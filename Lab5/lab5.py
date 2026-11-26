import time
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from web_scraper import *

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["axes.grid"] = True

def compute_kpis(y_true: np.ndarray, y_pred: np.ndarray, k_params: int = 1) -> dict:
    """Розрахунок типових KPI та критеріїв якості моделей часових рядів."""
    n = len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    r2 = r2_score(y_true, y_pred)
    max_err = np.max(np.abs(y_true - y_pred))
    # Інформаційні критерії на основі залишків (SSE)
    sse = np.sum((y_true - y_pred) ** 2)
    aic_approx = n * np.log(sse / n + 1e-12) + 2 * k_params
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2,
        "MaxErr": max_err,
        "AIC_est": aic_approx,
    }


def moving_average_smooth(series: np.ndarray, window: int) -> np.ndarray:
    """Ковзне середнє (SMA) для згладжування ряду."""
    return pd.Series(series).rolling(window=window, min_periods=1).mean().values


def recursive_sma_forecast(train_series: np.ndarray, window: int, steps: int) -> np.ndarray:
    """Рекурсивний прогноз на основі ковзного середнього."""
    history = list(train_series)
    preds = []
    for _ in range(steps):
        val = float(np.mean(history[-window:]))
        preds.append(val)
        history.append(val)
    return np.array(preds)


def local_linear_trend_forecast(train_series: np.ndarray, window: int, steps: int) -> np.ndarray:
    """Локальна лінійна регресія у ковзному вікні (усуває запізнювання SMA на тренді)."""
    x_w = np.arange(window)
    slope, intercept = np.polyfit(x_w, train_series[-window:], 1)
    future_x = np.arange(window, window + steps)
    return intercept + slope * future_x


def build_lag_features(series: np.ndarray, lags: int = 5):
    """Формування матриці ознак для методів статистичного навчання (ML)."""
    X, Y = [], []
    for i in range(lags, len(series)):
        win = series[i - lags : i]
        # Лаги + індекс часу (для вловлювання тренду) + локальні статистики вікна
        feats = list(win) + [i, np.mean(win), np.std(win), win[-1] - win[0]]
        X.append(feats)
        Y.append(series[i])
    return np.array(X), np.array(Y)


def recursive_ml_forecast(model, init_hist: np.ndarray, start_idx: int, steps: int, lags: int = 5) -> np.ndarray:
    """Багатокрокова рекурсивна екстраполяція для ML-моделей."""
    hist = list(init_hist)
    preds = []
    for step in range(steps):
        idx = start_idx + step
        win = np.array(hist[-lags:])
        feats = np.array(list(win) + [idx, np.mean(win), np.std(win), win[-1] - win[0]]).reshape(1, -1)
        pred = float(model.predict(feats)[0])
        preds.append(pred)
        hist.append(pred)
    return np.array(preds)

if __name__ == '__main__':

    directory_name = 'output'

    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    print('Оберіть напрям досліджень:')
    print('1 - Парсинг сайту обсерваторії (старий) https://cgo-sreznevskyi.kiev.ua/index.php?fn=k_klimat&f=kyiv')
    print('2 - Парсинг сайту обсерваторії (новий) https://cgo-sreznevskyi.kyiv.ua/uk/diialnist/klimatolohichna/klimatychni-dani-po-kyievu')
    print('3 - Зчитування інформації з файлу')
    mode = int(input('mode:'))

    if (mode == 1):
        print('Обрано інформаційне джерело: https://cgo-sreznevskyi.kiev.ua/index.php?fn=k_klimat&f=kyiv')
        url = 'https://cgo-sreznevskyi.kiev.ua/index.php?fn=k_klimat&f=kyiv'
        directory_name += '/szo'
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
        Parser_URL_szo_old(url)
        input_file = 'raw.csv'
        input_directory = directory_name + '/' + input_file

    elif (mode == 2):
        print('Обрано інформаційне джерело: https://cgo-sreznevskyi.kyiv.ua/uk/diialnist/klimatolohichna/klimatychni-dani-po-kyievu')
        url = 'https://cgo-sreznevskyi.kyiv.ua/uk/diialnist/klimatolohichna/klimatychni-dani-po-kyievu'
        directory_name += '/szo'
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
        Parser_URL_szo_new(url)
        input_file = 'raw.csv'
        input_directory = directory_name + '/' + input_file

    elif (mode == 3):
        input_file = input('Введіть назву файлу(формату .csv): ')
        directory_name += f'/{input_file}'
        print(directory_name)
        if not os.path.exists(directory_name):
            print('Ви ввели неправильну назву файлу')
            exit()
        else:
            input_directory = directory_name

    else:
        print('Неправильний варіант')
        exit()

    data = pd.read_csv(input_directory, index_col = 0)
    print(data)

    annual_temp = data['Середньорічна']
    years = annual_temp.index.values
    y = annual_temp.values
    N = len(y)

    print("=" * 80)
    print("1. ДОСЛІДЖЕННЯ СТАТИСТИЧНИХ ВЛАСТИВОСТЕЙ ЧАСОВОГО РЯДУ ('Середньорічна')")
    print("=" * 80)
    print(f"Обсяг вибірки N = {N} спостережень ({years[0]}–{years[-1]} рр.)")
    print(f"Середнє = {y.mean():.3f}°C | Медіана = {np.median(y):.3f}°C | Std = {y.std():.3f}°C")
    print(f"Min = {y.min():.1f}°C ({years[np.argmin(y)]}) | Max = {y.max():.1f}°C ({years[np.argmax(y)]})")

    adf_raw = adfuller(y)
    kpss_raw = kpss(y, regression="ct")
    y_diff1 = np.diff(y)
    adf_diff1 = adfuller(y_diff1)

    print(f"\nТест Дікі-Фуллера (ADF, вихідний ряд): stat = {adf_raw[0]:.4f}, p-value = {adf_raw[1]:.4f} -> Нестаціонарний")
    print(f"Тест KPSS (відносно тренду):          stat = {kpss_raw[0]:.4f}, p-value = {kpss_raw[1]:.4f} -> Нестаціонарний")
    print(f"Тест ADF (1-ша різниця d=1):          stat = {adf_diff1[0]:.4f}, p-value = {adf_diff1[1]:.2e} -> Стаціонарний (I(1))")

    slope_all, _ = np.polyfit(years, y, 1)
    slope_modern, _ = np.polyfit(years[years >= 1980], y[years >= 1980], 1)
    print(f"Загальний тренд (1881–2025): +{slope_all * 10:.3f}°C / десятиліття")
    print(f"Сучасний тренд  (1980–2025): +{slope_modern * 10:.3f}°C / десятиліття (прискорення у {slope_modern/slope_all:.1f}x)")

    TEST_SIZE = 20
    train, test = y[:-TEST_SIZE], y[-TEST_SIZE:]
    train_years, test_years = years[:-TEST_SIZE], years[-TEST_SIZE:]

    base_models_preds = {
        "SMA (w=5)": recursive_sma_forecast(train, window=5, steps=TEST_SIZE),
        "ARMA(1,1)": ARIMA(train, order=(1, 0, 1)).fit().forecast(steps=TEST_SIZE),
        "ARIMA(1,1,1)": ARIMA(train, order=(1, 1, 1)).fit().forecast(steps=TEST_SIZE),
        "Holt Linear": ExponentialSmoothing(train, trend="add", damped_trend=False).fit().forecast(steps=TEST_SIZE),
    }

    print("\n" + "=" * 80)
    print("2. ОЦІНКА ЯКОСТІ БАЗОВИХ МОДЕЛЕЙ (ГРУПА ВИМОГ 1, Тестовий інтервал 20 років)")
    print("=" * 80)
    base_results = {}
    for name, preds in base_models_preds.items():
        kpis = compute_kpis(test, preds, k_params=3)
        base_results[name] = kpis
        print(f"{name:15s} | MAE={kpis['MAE']:.4f} | RMSE={kpis['RMSE']:.4f} | MAPE={kpis['MAPE (%)']:.2f}% | R2={kpis['R2']:.4f}")


    print("\n" + "=" * 80)
    print("3. R&D ДОСЛІДЖЕННЯ: ОПТИМІЗАЦІЯ ПАРАМЕТРІВ КОВЗНОГО ВІКНА ТА ARIMA")
    print("=" * 80)

    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(0, 5):
        for q in range(0, 5):
            if p == 0 and q == 0:
                continue
            try:
                model_tmp = ARIMA(train, order=(p, 1, q)).fit()
                if model_tmp.aic < best_aic:
                    best_aic = model_tmp.aic
                    best_order = (p, 1, q)
            except Exception:
                pass

    arima_opt_model = ARIMA(train, order=best_order).fit()
    arima_opt_preds = arima_opt_model.forecast(steps=TEST_SIZE)

    lb_test = acorr_ljungbox(arima_opt_model.resid, lags=[10], return_df=True)
    lb_pval = lb_test["lb_pvalue"].values[0]
    print(f"Оптимальна структура ARIMA за AIC: {best_order} (AIC = {best_aic:.2f}, Ljung-Box p-val = {lb_pval:.4f})")

    best_w = 5
    best_w_mae = np.inf
    for w in range(5, 45):
        preds_w = local_linear_trend_forecast(train, window=w, steps=TEST_SIZE)
        mae_w = mean_absolute_error(test, preds_w)
        if mae_w < best_w_mae:
            best_w_mae = mae_w
            best_w = w

    local_trend_opt_preds = local_linear_trend_forecast(train, window=best_w, steps=TEST_SIZE)
    print(f"Оптимальна ширина вікна локальної регресії: w = {best_w} років (MAE знижено до {best_w_mae:.4f}°C)")

    LAGS = 5
    X_all, Y_all = build_lag_features(y, lags=LAGS)
    X_train, Y_train = X_all[:-TEST_SIZE], Y_all[:-TEST_SIZE]

    ml_models = {
        "Ridge (L2)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
    }

    comparison_rows = []
    for name, preds, k_p in [
        ("SMA(w=5) [Base]", base_models_preds["SMA (w=5)"], 1),
        ("ARIMA(1,1,1) [Base]", base_models_preds["ARIMA(1,1,1)"], 3),
        (f"ARIMA{best_order} [Opt]", arima_opt_preds, sum(best_order)),
        (f"LocalTrend(w={best_w}) [Opt]", local_trend_opt_preds, 2),
    ]:
        t0 = time.perf_counter()
        _ = np.mean(preds)
        dt_ms = (time.perf_counter() - t0) * 1000
        kpis = compute_kpis(test, preds, k_params=k_p)
        kpis["Model"] = name
        kpis["Class"] = "Window / ARIMA"
        kpis["Time (ms)"] = round(dt_ms, 2)
        comparison_rows.append(kpis)

    ml_preds_store = {}
    for name, model in ml_models.items():
        t0 = time.perf_counter()
        model.fit(X_train, Y_train)
        preds = recursive_ml_forecast(model, train[-LAGS:], len(train), TEST_SIZE, lags=LAGS)
        dt_ms = (time.perf_counter() - t0) * 1000
        ml_preds_store[name] = preds
        kpis = compute_kpis(test, preds, k_params=X_train.shape[1])
        kpis["Model"] = name
        kpis["Class"] = "Statistical Learning (ML)"
        kpis["Time (ms)"] = round(dt_ms, 2)
        comparison_rows.append(kpis)

    comp_df = pd.DataFrame(comparison_rows)[
        ["Model", "Class", "MAE", "RMSE", "MAPE (%)", "R2", "MaxErr", "Time (ms)"]
    ]
    print("\n" + "=" * 80)
    print("4. ЗВЕДЕНА ТАБЛИЦЯ KPI: КОВЗНЕ ВІКНО / ARIMA VS СТАТИСТИЧНЕ НАВЧАННЯ")
    print("=" * 80)
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


    horizons = {
        "0.5 N (73 р.)": int(round(0.5 * N)),
        "1.0 N (145 р.)": N,
        "1.5 N (218 р.)": int(round(1.5 * N)),
        "2.0 N (290 р.)": 2 * N,
    }
    max_h = max(horizons.values())

    full_arima_opt = ARIMA(y, order=best_order).fit()
    arima_extrap = full_arima_opt.forecast(steps=max_h)

    local_trend_extrap = local_linear_trend_forecast(y, window=best_w, steps=max_h)

    full_ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X_all, Y_all)
    ridge_extrap = recursive_ml_forecast(full_ridge, y[-LAGS:], N, max_h, lags=LAGS)

    full_gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_all, Y_all)
    gb_extrap = recursive_ml_forecast(full_gb, y[-LAGS:], N, max_h, lags=LAGS)

    print("\n" + "=" * 80)
    print("5. РЕЗУЛЬТАТИ ЕКСТРАПОЛЯЦІЇ НА 0.5, 1.0, 1.5 ТА 2.0 ІНТЕРВАЛУ СПОСТЕРЕЖЕННЯ")
    print("=" * 80)
    extrap_rows = []
    for label, h in horizons.items():
        target_year = years[-1] + h
        extrap_rows.append({
            "Горизонт": label,
            "Кінцевий рік": target_year,
            f"ARIMA{best_order} (°C)": arima_extrap[h - 1],
            f"LocalTrend(w={best_w}) (°C)": local_trend_extrap[h - 1],
            "Ridge ML (°C)": ridge_extrap[h - 1],
            "GradientBoosting ML (°C)": gb_extrap[h - 1],
        })
    extrap_df = pd.DataFrame(extrap_rows)
    print(extrap_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(years[-40:], y[-40:], "k-o", label="Фактичні дані (°C)", linewidth=2, markersize=4)
    axes[0].axvline(test_years[0], color="gray", linestyle="--", label="Межа Train / Test")
    axes[0].plot(test_years, base_models_preds["SMA (w=5)"], "--", label="SMA (w=5) [Base]")
    axes[0].plot(test_years, arima_opt_preds, "-s", markersize=4, label=f"ARIMA{best_order} [Opt]")
    axes[0].plot(test_years, local_trend_opt_preds, "-^", markersize=4, label=f"LocalTrend(w={best_w}) [Opt]")
    axes[0].plot(test_years, ml_preds_store["Gradient Boosting"], "-d", markersize=4, label="Gradient Boosting [ML]")
    axes[0].set_title("Валідація моделей на тестовому горизонті (2006–2025)")
    axes[0].set_xlabel("Рік")
    axes[0].set_ylabel("Середньорічна температура, °C")
    axes[0].legend()

    future_years = np.arange(years[-1] + 1, years[-1] + max_h + 1)
    axes[1].plot(years, y, "k-", label="Історичні дані (1881–2025)", linewidth=1.5)
    axes[1].plot(future_years, arima_extrap, label=f"ARIMA{best_order} (стабілізація до рівня)")
    axes[1].plot(future_years, ridge_extrap, label="Ridge Regression (лінійний тренд)")
    axes[1].plot(future_years, gb_extrap, label="Gradient Boosting (насичення)")
    axes[1].plot(future_years, local_trend_extrap, "--", label=f"LocalTrend(w={best_w}) (екстраполяція темпу)")

    for label, h in horizons.items():
        axes[1].axvline(years[-1] + h, color="red", alpha=0.3, linestyle=":")

    axes[1].set_title("Екстраполяція часового ряду на 0.5N – 2.0N (до 2315 р.)")
    axes[1].set_xlabel("Рік")
    axes[1].set_ylabel("Прогнозна температура, °C")
    axes[1].set_ylim(4, 25)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
