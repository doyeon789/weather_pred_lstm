#pip install pandas supabase scikit-learn
#pip install torch --index-url https://download.pytorch.org/whl/cpu
#pip install numpy==1.24.3
#pip install scipy==1.10.1

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from supabase import create_client
from datetime import timedelta, datetime, date
from sklearn.preprocessing import MinMaxScaler

# --- 1. Supabase 연결 ---
url = "https://vcqqokmyyjsvxyvuzgmv.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZjcXFva215eWpzdnh5dnV6Z212Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA5MjE2OTgsImV4cCI6MjA3NjQ5NzY5OH0.lv0mtev8N61_QicEObv5Bdbk7Gpwnh-tLnkX0M-SI5Q"
supabase = create_client(url, key)

# --- 2. 과거 데이터 불러오기 ---
past_resp = supabase.table("r_weather_data").select("*").execute()
past_data = pd.DataFrame(past_resp.data)

# --- 3. 전처리 ---
past_data['datetime'] = pd.to_datetime(past_data['r_timestamp'])
past_data['r_insolation'] = past_data['r_insolation'].replace(-9, 0)
past_data = past_data.rename(columns={'r_insolation': 'target'})
past_data = past_data.dropna(subset=['target'])

# --- 4. 함수 정의 ---
def get_time_window_avg(df, current_time, window_hours=2):
    """current_time 기준 ±window_hours 시간 범위 내 target 평균 반환"""
    start = current_time - timedelta(hours=window_hours)
    end = current_time + timedelta(hours=window_hours)
    window_data = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
    if window_data.empty:
        return None
    return window_data['target'].mean()

def create_custom_sequence(df, pred_time, seq_length=13, intervals=[24,48,72], window_hours=2):
    """pred_time 기준으로 intervals 시간만큼 과거 ±window_hours 구간 평균값으로 시퀀스 생성"""
    seq = []
    for hours_ago in intervals:
        base_time = pred_time - timedelta(hours=hours_ago)
        val = get_time_window_avg(df, base_time, window_hours)
        if val is None:
            val = 0.0
        seq.append(val)
    seq = (seq * (seq_length // len(seq) + 1))[:seq_length]
    return np.array(seq)

def create_training_data(df, seq_length=13, intervals=[24,48,72], window_hours=2):
    X, y, times = [], [], []
    df = df.sort_values('datetime').reset_index(drop=True)
    for idx in range(seq_length, len(df)):
        pred_time = df.loc[idx, 'datetime']
        seq = create_custom_sequence(df, pred_time, seq_length, intervals, window_hours)
        target = df.loc[idx, 'target']
        X.append(seq)
        y.append(target)
        times.append(pred_time)
    return np.array(X), np.array(y), times

# --- 5. 학습 데이터 생성 ---
X_train, y_train, train_times = create_training_data(past_data)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

# --- 6. LSTM 모델 정의 ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 7. 모델 학습 ---
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- 8. 24시간 예측 ---
model.eval()
predictions = []
with torch.no_grad():
    last_time = past_data['datetime'].max().replace(minute=0, second=0, microsecond=0)
    for hour in range(24):
        pred_time = last_time.replace(hour=hour)
        seq = create_custom_sequence(past_data, pred_time, seq_length=13)
        x_input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model(x_input).item()
        pred = max(pred, 0)
        if pred_time.hour < 6 or pred_time.hour > 18:
            pred = 0.0
        predictions.append((pred_time, pred))

# --- 9. 전력 소비 계산 ---
def calculate_power(irradiance_pred):
    max_irradiance = 1000
    base_power = 0.6
    led_power = 5 * 0.53
    rgb_power = 4 * 0.53
    servo_power = (2 * 1.53)/50
    sensor_power = 0.3
    power = base_power + led_power * (irradiance_pred / max_irradiance) + rgb_power + servo_power + sensor_power
    return power


# --- 10. 기존 데이터 삭제 후 새 데이터 업로드 ---
records = []
for pred_time, val in predictions:
    power = calculate_power(val)
    record = {
        "predicted_time": pred_time.isoformat(),
        "pred_insolation": float(val),
        "pred_power": float(power),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    records.append(record)

try:
    # 기존 모든 prediction 데이터 삭제
    supabase.table("prediction").delete().not_.is_("prediction_id",None).execute()
    
    print("🗑️ 기존 prediction 테이블의 모든 데이터 삭제 완료")

    # 새 데이터 삽입
    response = supabase.table("prediction").insert(records).execute()
    print(f"\n✅ Supabase 업로드 완료: {len(records)}개의 예측값 저장됨")

except Exception as e:
    print("❌ Supabase 작업 중 오류 발생:", e)



# --- 11. 콘솔 출력 ---
print("\n예측된 24시간 일사량 및 전력 소비량:")
for pred_time, val in predictions:
    power = calculate_power(val)
    print(f"{pred_time.strftime('%Y-%m-%d %H:%M')} → 일사량 {val:.3f} W/m² | 소비전력 {power:.3f} W")

print("\n✅ 전체 프로세스 완료")
