"""
動的プライシングの実装
需要予測モデルを使った動的プライシングシステム
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 日本語フォントの設定
try:
    # macOSの場合
    matplotlib.rcParams['font.family'] = 'Hiragino Sans'
except:
    try:
        # Linuxの場合
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
    except:
        # フォントが見つからない場合は警告を無視
        pass


def generate_historical_data():
    """過去の販売データをシミュレート"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

    # 需要データ（価格が低いほど需要が高い）
    base_demand = 100
    price_elasticity = -1.5  # 価格弾力性

    data = []
    for date in dates:
        # 曜日効果（週末は需要が高い）
        weekday_factor = 1.2 if date.weekday() >= 5 else 1.0
        
        # ランダムな価格（1000円〜3000円）
        price = np.random.uniform(1000, 3000)
        
        # 需要 = ベース需要 × 価格弾力性 × 曜日効果 × ランダムノイズ
        demand = base_demand * (price / 2000) ** price_elasticity * weekday_factor * np.random.uniform(0.8, 1.2)
        demand = max(0, int(demand))  # 需要は0以上
        
        data.append({
            'date': date,
            'price': price,
            'demand': demand,
            'revenue': price * demand
        })

    historical_data = pd.DataFrame(data)
    return historical_data


def build_demand_model(historical_data):
    """需要予測モデルを構築"""
    # 価格と需要の関係を学習
    X = historical_data[['price']].values
    y = historical_data['demand'].values

    # データを学習用（80%）と検証用（20%）に分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 多項式特徴量を使用（価格と需要の関係は非線形の可能性があるため）
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])

    # 学習データでモデルを学習
    model.fit(X_train, y_train)

    # 学習データと検証データの両方で予測精度を確認
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    print(f"学習データでの予測精度 (R²): {train_score:.3f}")
    print(f"検証データでの予測精度 (R²): {val_score:.3f}")

    # 過学習のチェック
    if train_score - val_score > 0.1:
        print("警告: 学習データと検証データの精度に大きな差があります。過学習の可能性があります。")
    else:
        print("学習データと検証データの精度が近いため、モデルは適切に汎化できています。")
    
    return model


def calculate_optimal_price(model, price_range=(500, 4000), step=10):
    """
    収益を最大化する価格を計算
    
    Args:
        model: 需要予測モデル
        price_range: 価格の探索範囲
        step: 価格の刻み幅
    
    Returns:
        最適価格とその時の予測需要、予測収益
    """
    prices = np.arange(price_range[0], price_range[1], step)
    optimal_price = None
    max_revenue = 0
    optimal_demand = 0
    
    for price in prices:
        predicted_demand = model.predict([[price]])[0]
        predicted_demand = max(0, predicted_demand)  # 需要は0以上
        revenue = price * predicted_demand
        
        if revenue > max_revenue:
            max_revenue = revenue
            optimal_price = price
            optimal_demand = predicted_demand
    
    return optimal_price, optimal_demand, max_revenue


def dynamic_pricing_with_inventory(model, current_inventory, target_inventory=50, base_price=None):
    """
    在庫を考慮した動的プライシング
    
    Args:
        model: 需要予測モデル
        current_inventory: 現在の在庫数
        target_inventory: 目標在庫数
        base_price: ベース価格（Noneの場合は最適価格を計算）
    
    Returns:
        調整後の価格
    """
    if base_price is None:
        base_price, _, _ = calculate_optimal_price(model)
    
    # 在庫調整係数
    inventory_ratio = current_inventory / target_inventory
    
    if inventory_ratio > 1.5:  # 在庫が目標の1.5倍以上
        # 在庫が多いので価格を下げて需要を喚起
        adjustment_factor = 0.9
    elif inventory_ratio < 0.5:  # 在庫が目標の半分以下
        # 在庫が少ないので価格を上げる
        adjustment_factor = 1.1
    else:
        adjustment_factor = 1.0
    
    adjusted_price = base_price * adjustment_factor
    return adjusted_price


def dynamic_pricing_with_time(model, current_time, base_price=None):
    """
    時間帯を考慮した動的プライシング
    
    Args:
        model: 需要予測モデル
        current_time: 現在の日時
        base_price: ベース価格
    
    Returns:
        調整後の価格
    """
    if base_price is None:
        base_price, _, _ = calculate_optimal_price(model)
    
    # 曜日効果
    if current_time.weekday() >= 5:  # 週末
        time_factor = 1.1  # 週末は需要が高いので価格を上げる
    else:
        time_factor = 1.0
    
    # 時間帯効果（例: 夜間は需要が低い）
    hour = current_time.hour
    if 9 <= hour <= 18:  # 営業時間
        hour_factor = 1.0
    else:
        hour_factor = 0.95  # 営業時間外は価格を少し下げる
    
    adjusted_price = base_price * time_factor * hour_factor
    return adjusted_price


def visualize_pricing(model, optimal_price):
    """価格と需要、収益の関係を可視化"""
    # 価格と需要の関係を可視化
    prices = np.arange(500, 4000, 50)
    demands = [max(0, model.predict([[p]])[0]) for p in prices]
    revenues = [p * d for p, d in zip(prices, demands)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 価格と需要の関係
    ax1.plot(prices, demands, label='予測需要')
    ax1.axvline(optimal_price, color='r', linestyle='--', label=f'最適価格 ({optimal_price:.0f}円)')
    ax1.set_xlabel('価格 (円)')
    ax1.set_ylabel('需要 (個)')
    ax1.set_title('価格と需要の関係')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 価格と収益の関係
    ax2.plot(prices, revenues, label='予測収益')
    ax2.axvline(optimal_price, color='r', linestyle='--', label=f'最適価格 ({optimal_price:.0f}円)')
    ax2.set_xlabel('価格 (円)')
    ax2.set_ylabel('収益 (円)')
    ax2.set_title('価格と収益の関係')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pricing_analysis.png', dpi=150, bbox_inches='tight')
    print("可視化結果を 'pricing_analysis.png' に保存しました。")
    plt.show()


def main():
    """メイン処理"""
    print("=" * 60)
    print("動的プライシングシステム")
    print("=" * 60)
    
    # データの準備
    print("\n[1] 過去の販売データを生成中...")
    historical_data = generate_historical_data()
    print(historical_data.head())
    
    # 需要予測モデルの構築
    print("\n[2] 需要予測モデルを構築中...")
    model = build_demand_model(historical_data)
    
    # 最適価格の計算
    print("\n[3] 最適価格を計算中...")
    optimal_price, optimal_demand, optimal_revenue = calculate_optimal_price(model)
    print(f"最適価格: {optimal_price:.0f}円")
    print(f"予測需要: {optimal_demand:.1f}個")
    print(f"予測収益: {optimal_revenue:.0f}円")
    
    # 在庫を考慮した動的プライシング
    print("\n[4] 在庫を考慮した価格調整...")
    low_inventory_price = dynamic_pricing_with_inventory(model, current_inventory=10, target_inventory=50)
    print(f"在庫が少ない場合の価格: {low_inventory_price:.0f}円")
    
    high_inventory_price = dynamic_pricing_with_inventory(model, current_inventory=100, target_inventory=50)
    print(f"在庫が多い場合の価格: {high_inventory_price:.0f}円")
    
    # 時間帯を考慮した動的プライシング
    print("\n[5] 時間帯を考慮した価格調整...")
    weekend_price = dynamic_pricing_with_time(model, datetime(2024, 1, 6, 14, 0, 0))
    print(f"週末の営業時間の価格: {weekend_price:.0f}円")
    
    weekday_night_price = dynamic_pricing_with_time(model, datetime(2024, 1, 3, 20, 0, 0))
    print(f"平日の夜間の価格: {weekday_night_price:.0f}円")
    
    # 可視化
    print("\n[6] 可視化中...")
    visualize_pricing(model, optimal_price)
    
    print("\n" + "=" * 60)
    print("処理が完了しました。")
    print("=" * 60)


if __name__ == "__main__":
    main()
