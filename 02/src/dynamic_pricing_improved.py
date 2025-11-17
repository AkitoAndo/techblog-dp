"""
動的プライシングの改善実装
正則化、可視化の改善、エラーハンドリングを追加した動的プライシングシステム
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


def generate_historical_data(days=90):
    """
    過去の販売データをシミュレート
    
    Args:
        days: 生成するデータの日数（デフォルト: 90日）
    
    Returns:
        historical_data: 過去の販売データを含むDataFrame
    """
    try:
        if days <= 0:
            raise ValueError("日数は1以上である必要があります。")
        
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

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
    
    except Exception as e:
        print(f"データ生成中にエラーが発生しました: {e}")
        raise


def build_demand_model(historical_data, alpha=1.0, degree=2):
    """
    需要予測モデルを構築（正則化付き）
    
    Args:
        historical_data: 過去の販売データ
        alpha: Ridge回帰の正則化パラメータ（デフォルト: 1.0）
        degree: 多項式特徴量の次数（デフォルト: 2）
    
    Returns:
        model: 学習済みのモデル
        X_train: 学習データの特徴量
        y_train: 学習データの目的変数
        X_val: 検証データの特徴量
        y_val: 検証データの目的変数
    """
    try:
        if historical_data is None or len(historical_data) == 0:
            raise ValueError("データが空です。")
        
        if alpha < 0:
            raise ValueError("正則化パラメータは0以上である必要があります。")
        
        # 価格と需要の関係を学習
        X = historical_data[['price']].values
        y = historical_data['demand'].values

        # データを学習用（80%）と検証用（20%）に分割
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 正則化を導入したモデル（Ridge回帰）
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('ridge', Ridge(alpha=alpha))
        ])

        # 学習データでモデルを学習
        model.fit(X_train, y_train)

        # 学習データと検証データの両方で予測精度を確認
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        print(f"学習データでの予測精度 (R²): {train_score:.3f}")
        print(f"検証データでの予測精度 (R²): {val_score:.3f}")

        # MAEとRMSEも計算
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        print(f"\n学習データでのMAE: {train_mae:.2f}")
        print(f"検証データでのMAE: {val_mae:.2f}")
        print(f"学習データでのRMSE: {train_rmse:.2f}")
        print(f"検証データでのRMSE: {val_rmse:.2f}")

        # 過学習のチェック
        if train_score - val_score > 0.1:
            print("\n警告: 学習データと検証データの精度に大きな差があります。過学習の可能性があります。")
        else:
            print("\n学習データと検証データの精度が近いため、モデルは適切に汎化できています。")
        
        return model, X_train, y_train, X_val, y_val
    
    except Exception as e:
        print(f"モデル構築中にエラーが発生しました: {e}")
        raise


def calculate_optimal_price(model, price_range=(500, 4000), step=10):
    """
    収益を最大化する価格を計算
    
    Args:
        model: 需要予測モデル
        price_range: 価格の探索範囲（最小値, 最大値）
        step: 価格の刻み幅
    
    Returns:
        最適価格とその時の予測需要、予測収益
    
    Raises:
        ValueError: 価格範囲が無効な場合
    """
    try:
        min_price, max_price = price_range
        if min_price >= max_price or min_price < 0:
            raise ValueError("価格範囲が無効です。最小価格は最大価格より小さく、0以上である必要があります。")
        
        prices = np.arange(min_price, max_price, step)
        if len(prices) == 0:
            raise ValueError("価格の探索範囲が狭すぎます。")
        
        optimal_price = None
        max_revenue = 0
        optimal_demand = 0
        
        for price in prices:
            try:
                predicted_demand = model.predict([[price]])[0]
                predicted_demand = max(0, predicted_demand)  # 需要は0以上
                revenue = price * predicted_demand
                
                if revenue > max_revenue:
                    max_revenue = revenue
                    optimal_price = price
                    optimal_demand = predicted_demand
            except Exception as e:
                print(f"価格 {price} での予測中にエラーが発生しました: {e}")
                continue
        
        if optimal_price is None:
            raise ValueError("最適価格が見つかりませんでした。")
        
        return optimal_price, optimal_demand, max_revenue
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


def dynamic_pricing_with_inventory(model, current_inventory, target_inventory=50, base_price=None):
    """
    在庫を考慮した動的プライシング
    
    Args:
        model: 需要予測モデル
        current_inventory: 現在の在庫数（0以上）
        target_inventory: 目標在庫数（0より大きい値）
        base_price: ベース価格（Noneの場合は最適価格を計算）
    
    Returns:
        調整後の価格
    
    Raises:
        ValueError: 在庫数が無効な場合
    """
    try:
        if current_inventory < 0:
            raise ValueError("在庫数は0以上である必要があります。")
        if target_inventory <= 0:
            raise ValueError("目標在庫数は0より大きい値である必要があります。")
        
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
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


def dynamic_pricing_with_time(model, current_time, base_price=None):
    """
    時間帯を考慮した動的プライシング
    
    Args:
        model: 需要予測モデル
        current_time: 現在の日時（datetimeオブジェクト）
        base_price: ベース価格
    
    Returns:
        調整後の価格
    
    Raises:
        ValueError: 日時が無効な場合
    """
    try:
        if not isinstance(current_time, datetime):
            raise ValueError("current_timeはdatetimeオブジェクトである必要があります。")
        
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
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


def visualize_pricing(model, optimal_price, X_train, y_train, X_val, y_val):
    """
    価格と需要、収益の関係、および残差プロットを可視化
    
    Args:
        model: 需要予測モデル
        optimal_price: 最適価格
        X_train: 学習データの特徴量
        y_train: 学習データの目的変数
        X_val: 検証データの特徴量
        y_val: 検証データの目的変数
    """
    try:
        # 価格と需要の関係を可視化
        prices = np.arange(500, 4000, 50)
        demands = [max(0, model.predict([[p]])[0]) for p in prices]
        revenues = [p * d for p, d in zip(prices, demands)]
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 価格と需要の関係
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(prices, demands, label='Predicted Demand', linewidth=2)
        ax1.axvline(optimal_price, color='r', linestyle='--', label=f'Optimal Price ({optimal_price:.0f} yen)')
        ax1.set_xlabel('Price (yen)')
        ax1.set_ylabel('Demand (units)')
        ax1.set_title('Price vs Demand')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 価格と収益の関係
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(prices, revenues, label='Predicted Revenue', linewidth=2, color='green')
        ax2.axvline(optimal_price, color='r', linestyle='--', label=f'Optimal Price ({optimal_price:.0f} yen)')
        ax2.set_xlabel('Price (yen)')
        ax2.set_ylabel('Revenue (yen)')
        ax2.set_title('Price vs Revenue')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 学習データの残差プロット
        ax3 = plt.subplot(2, 3, 3)
        train_pred = model.predict(X_train)
        train_residuals = y_train - train_pred
        ax3.scatter(train_pred, train_residuals, alpha=0.5, color='blue')
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Training Data Residuals')
        ax3.grid(True, alpha=0.3)
        
        # 4. 検証データの残差プロット
        ax4 = plt.subplot(2, 3, 4)
        val_pred = model.predict(X_val)
        val_residuals = y_val - val_pred
        ax4.scatter(val_pred, val_residuals, alpha=0.5, color='orange')
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Validation Data Residuals')
        ax4.grid(True, alpha=0.3)
        
        # 5. 学習データの予測値 vs 実際の値
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(y_train, train_pred, alpha=0.5, color='blue')
        min_val = min(min(y_train), min(train_pred))
        max_val = max(max(y_train), max(train_pred))
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax5.set_xlabel('Actual')
        ax5.set_ylabel('Predicted')
        ax5.set_title('Training Data: Actual vs Predicted')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 検証データの予測値 vs 実際の値
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(y_val, val_pred, alpha=0.5, color='orange')
        min_val = min(min(y_val), min(val_pred))
        max_val = max(max(y_val), max(val_pred))
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax6.set_xlabel('Actual')
        ax6.set_ylabel('Predicted')
        ax6.set_title('Validation Data: Actual vs Predicted')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pricing_analysis_improved.png', dpi=150, bbox_inches='tight')
        print("可視化結果を 'pricing_analysis_improved.png' に保存しました。")
        plt.show()
    
    except Exception as e:
        print(f"可視化中にエラーが発生しました: {e}")
        raise


def main():
    """メイン処理"""
    print("=" * 60)
    print("動的プライシングシステム（改善版）")
    print("=" * 60)
    
    try:
        # データの準備
        print("\n[1] 過去90日分の販売データを生成中...")
        historical_data = generate_historical_data(days=90)
        print(f"データ数: {len(historical_data)}日分")
        print(historical_data.head())
        
        # 需要予測モデルの構築
        print("\n[2] 需要予測モデルを構築中（正則化付き）...")
        model, X_train, y_train, X_val, y_val = build_demand_model(historical_data, alpha=1.0)
        
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
        print("\n[6] 可視化中（残差プロット含む）...")
        visualize_pricing(model, optimal_price, X_train, y_train, X_val, y_val)
        
        print("\n" + "=" * 60)
        print("処理が完了しました。")
        print("=" * 60)
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
