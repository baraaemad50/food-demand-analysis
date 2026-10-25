"""Evaluate and train a Random Forest food-demand model.

Validation MAE is calculated on the final ten labeled training weeks.
Predictions for test.csv are exported without claiming test accuracy because
test.csv does not contain actual num_orders values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "num_orders"
KEYS = ["week", "center_id", "meal_id"]
CATEGORICAL_FEATURES = [
    "center_id",
    "meal_id",
    "category",
    "cuisine",
    "emailer_for_promotion",
    "homepage_featured",
]
NUMERIC_FEATURES = ["week", "checkout_price", "base_price"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def log_row_count(data: pd.DataFrame, operation: str) -> None:
    """Print row count and demand sum when the target exists."""
    if TARGET in data.columns:
        print(
            f"{operation}: الصفوف = {len(data):,} | "
            f"مجموع num_orders = {data[TARGET].sum():,.0f}"
        )
    else:
        print(
            f"{operation}: الصفوف = {len(data):,} | "
            "مجموع num_orders غير متاح في هذه البيانات"
        )


def load_and_enrich(
    csv_path: Path,
    meal_info: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Load a raw CSV and join meal labels using a validated many-to-one merge."""
    data = pd.read_csv(csv_path)
    required = {
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
    }
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing_columns)}"
        )
    if source_name == "train.csv" and TARGET not in data.columns:
        raise ValueError("train.csv must contain num_orders for model training.")

    log_row_count(data, f"قراءة {source_name}")
    before_rows = len(data)
    before_orders = data[TARGET].sum() if TARGET in data.columns else None
    enriched = data.merge(
        meal_info[["meal_id", "category", "cuisine"]],
        on="meal_id",
        how="left",
        validate="m:1",
        indicator="_meal_match",
    )
    log_row_count(enriched, f"بعد دمج بيانات الوجبات مع {source_name}")

    if len(enriched) != before_rows:
        raise AssertionError(f"Meal merge changed the {source_name} row count.")
    if TARGET in enriched.columns and enriched[TARGET].sum() != before_orders:
        raise AssertionError(f"Meal merge changed the {source_name} demand sum.")
    unmatched = enriched["_meal_match"].ne("both")
    if unmatched.any():
        raise ValueError(
            f"{source_name} contains {int(unmatched.sum()):,} unmatched meal_id rows."
        )
    return enriched.drop(columns="_meal_match")


def make_model(n_estimators: int, random_state: int) -> Pipeline:
    """Create a sparse one-hot encoded Random Forest regression pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=16,
        min_samples_leaf=10,
        max_features="sqrt",
        max_samples=0.75,
        n_jobs=4,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("random_forest", forest),
        ]
    )


def report_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute mean absolute error without an additional metrics dependency."""
    if len(actual) == 0:
        raise ValueError("Cannot calculate MAE on an empty validation set.")
    return float((actual - predicted).abs().mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Random Forest and predict demand for test.csv."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dataraw",
        help="Directory containing train.csv, test.csv, and meal_info.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports",
        help="Directory for CSV, markdown, and figure outputs.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees per Random Forest.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be positive.")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    meal_path = args.data_dir / "meal_info.csv"
    for csv_path in [train_path, test_path, meal_path]:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Required input file does not exist: {csv_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # نقرأ جدول تعريف الوجبات ونتحقق من فرادة مفتاح الدمج.
    meal_info = pd.read_csv(meal_path)
    if not {"meal_id", "category", "cuisine"}.issubset(meal_info.columns):
        raise ValueError("meal_info.csv must contain meal_id, category, and cuisine.")
    if meal_info["meal_id"].duplicated().any():
        raise ValueError("meal_info.csv contains duplicate meal_id keys.")

    # نقرأ بيانات التدريب والاختبار ونضيف تصنيفات الوجبات.
    train_data = load_and_enrich(train_path, meal_info, "train.csv")
    test_data = load_and_enrich(test_path, meal_info, "test.csv")
    if "id" not in test_data.columns:
        raise ValueError("test.csv must contain id to preserve prediction-row identity.")
    if TARGET in test_data.columns:
        raise ValueError("test.csv unexpectedly contains num_orders; check its schema.")
    if train_data[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Training features or target contain missing values.")
    if test_data[FEATURES].isna().any().any():
        raise ValueError("Test features contain missing values.")

    # نحجز آخر عشرة أسابيع معلّمة للتحقق الزمني دون خلط عشوائي.
    ordered_weeks = sorted(train_data["week"].unique())
    if len(ordered_weeks) <= 10:
        raise ValueError("At least eleven distinct training weeks are required.")
    validation_weeks = ordered_weeks[-10:]
    validation_start = validation_weeks[0]
    if test_data["week"].min() <= train_data["week"].max():
        raise ValueError("test.csv weeks must follow the labeled training weeks.")
    model_train = train_data.loc[train_data["week"] < validation_start].copy()
    validation_data = train_data.loc[
        train_data["week"].isin(validation_weeks)
    ].copy()
    log_row_count(model_train, "بعد تصفية أسابيع التدريب للتحقق الزمني")
    log_row_count(validation_data, "بعد تصفية آخر عشرة أسابيع للتحقق")
    if model_train["week"].max() >= validation_data["week"].min():
        raise AssertionError("Temporal validation overlaps with training weeks.")

    # ندرب النموذج على الماضي فقط ثم نتنبأ بأسابيع التحقق.
    validation_model = make_model(args.n_estimators, args.random_state)
    validation_model.fit(model_train[FEATURES], model_train[TARGET])
    validation_predictions = validation_data[
        KEYS + [TARGET]
    ].copy()
    validation_predictions["rf_prediction"] = validation_model.predict(
        validation_data[FEATURES]
    ).clip(min=0)
    validation_predictions["rf_prediction"] = validation_predictions[
        "rf_prediction"
    ].round()

    # نربط كل سجل تحقق بطلب الوجبة والمركز في الأسبوع السابق.
    previous_week_orders = train_data[KEYS + [TARGET]].copy()
    previous_week_orders["week"] += 1
    previous_week_orders = previous_week_orders.rename(
        columns={TARGET: "previous_week_orders"}
    )
    before_rows = len(validation_predictions)
    before_orders = validation_predictions[TARGET].sum()
    validation_predictions = validation_predictions.merge(
        previous_week_orders,
        on=KEYS,
        how="left",
        validate="1:1",
    )
    log_row_count(
        validation_predictions,
        "بعد ربط نتائج التحقق بطلب الأسبوع السابق",
    )
    if len(validation_predictions) != before_rows:
        raise AssertionError("Previous-week merge changed validation row count.")
    if validation_predictions[TARGET].sum() != before_orders:
        raise AssertionError("Previous-week merge changed validation demand.")

    # نقيس MAE للنموذج على كامل فترة التحقق.
    rf_validation_mae = report_mae(
        validation_predictions[TARGET],
        validation_predictions["rf_prediction"],
    )
    print(f"Random Forest MAE على آخر عشرة أسابيع تدريبية: {rf_validation_mae:,.2f}")

    # نقارن النموذج بخط أساس الأسبوع السابق على السجلات المتطابقة فقط.
    matched_validation = validation_predictions.loc[
        validation_predictions["previous_week_orders"].notna()
    ].copy()
    log_row_count(
        matched_validation,
        "بعد تصفية سجلات التحقق ذات الطلب المرصود في الأسبوع السابق",
    )
    matched_rf_mae = report_mae(
        matched_validation[TARGET],
        matched_validation["rf_prediction"],
    )
    previous_week_mae = report_mae(
        matched_validation[TARGET],
        matched_validation["previous_week_orders"],
    )
    matched_coverage = len(matched_validation) / len(validation_predictions) * 100
    print(
        f"MAE Random Forest على السجلات المتطابقة: {matched_rf_mae:,.2f} | "
        f"MAE خط أساس الأسبوع السابق: {previous_week_mae:,.2f} | "
        f"تغطية خط الأساس: {matched_coverage:.2f}%"
    )
    if matched_rf_mae > previous_week_mae:
        print(
            "تنبيه: خط أساس الأسبوع السابق أقل خطأً على السجلات المتطابقة؛ "
            "لا تستبدله بنموذج Random Forest قبل تحسين التحقق."
        )

    # نعيد تدريب النموذج على كامل السجل المعلّم للتنبؤ بصفوف test.csv.
    final_model = make_model(args.n_estimators, args.random_state)
    final_model.fit(train_data[FEATURES], train_data[TARGET])
    test_predictions = test_data[
        ["id", "week", "center_id", "meal_id", "category", "cuisine"]
    ].copy()
    test_predictions["predicted_num_orders"] = (
        final_model.predict(test_data[FEATURES]).clip(min=0).round()
    )
    if len(test_predictions) != len(test_data):
        raise AssertionError("Prediction row count does not match test.csv.")
    log_row_count(test_predictions, "تنبؤات test.csv")
    predictions_path = args.output_dir / "random_forest_test_predictions.csv"
    test_predictions.to_csv(predictions_path, index=False)
    print(f"تم حفظ التنبؤات: {predictions_path}")
    print(
        "لا يمكن حساب MAE على test.csv لعدم توفر num_orders الفعلية؛ "
        "مقاييس MAE أعلاه من تحقق زمني تاريخي فقط."
    )

    # نلخص الطلب المتوقع حسب الفئة للإجابة عن أولوية الشراء.
    category_forecast = (
        test_predictions.groupby("category", as_index=False)
        .agg(
            forecast_orders=("predicted_num_orders", "sum"),
            forecast_rows=("predicted_num_orders", "size"),
        )
        .sort_values("forecast_orders", ascending=False)
    )
    category_forecast["forecast_share_pct"] = (
        category_forecast["forecast_orders"]
        / category_forecast["forecast_orders"].sum()
        * 100
    )
    top_categories = category_forecast.head(10).copy()
    top_share = float(top_categories["forecast_share_pct"].sum())
    top_category = top_categories.iloc[0]
    top_pair = (
        test_predictions.groupby(["meal_id", "center_id"], as_index=False)[
            "predicted_num_orders"
        ]
        .sum()
        .sort_values("predicted_num_orders", ascending=False)
        .iloc[0]
    )

    # نرسم الفئات العشر ذات أعلى طلب متوقع خلال أفق الاختبار.
    fig, ax = plt.subplots(figsize=(10, 6))
    chart_data = top_categories.sort_values("forecast_orders", ascending=True)
    ax.barh(
        chart_data["category"],
        chart_data["forecast_orders"],
        color="#4C72B0",
    )
    ax.set_title("أعلى الفئات في الطلب المتوقع عبر أسابيع test.csv")
    ax.set_xlabel("إجمالي الطلب المتوقع")
    ax.set_ylabel("الفئة")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    chart_path = figures_dir / "random_forest_category_forecast.png"
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"تم حفظ الرسم: {chart_path}")

    # نكتب ملخصًا عربيًا جاهزًا للإضافة إلى README.
    insight_path = args.output_dir / "random_forest_client_insight.md"
    insight_path.write_text(
        "\n".join(
            [
                "## استخدام توقعات الطلب في التخطيط للشراء",
                "",
                f"### سؤال العميل: أي الفئات أولى بالمراجعة الشرائية خلال أسابيع "
                f"`test.csv` ({int(test_data['week'].min())}–"
                f"{int(test_data['week'].max())})؟",
                "",
                f"قدّر نموذج Random Forest أن فئة **{top_category['category']}** "
                f"ستسجل أعلى إجمالي طلب متوقع، بنحو "
                f"**{top_category['forecast_orders']:,.0f}** طلب عبر صفوف الاختبار "
                f"({top_category['forecast_share_pct']:.1f}% من الإجمالي المتوقع). "
                f"وتضم أعلى عشر فئات مجتمعةً نحو **{top_share:.1f}%** من الطلب "
                "المتوقع. أعلى زوج وجبة/مركز في الجدول هو "
                f"`meal_id={int(top_pair['meal_id'])}`, "
                f"`center_id={int(top_pair['center_id'])}`، بإجمالي متوقع "
                f"**{top_pair['predicted_num_orders']:,.0f}** طلب.",
                "",
                f"![توقعات Random Forest حسب الفئة](figures/"
                f"{chart_path.name})",
                "",
                "استخدم هذه التقديرات لترتيب مراجعة الشراء والتوافر أولًا، "
                "ثم حدّد الكميات بعد تطبيق حجم العبوة ومهلة التوريد ومستوى "
                "الخدمة والمخزون الحالي والهدر المتوقع. التوقع ليس أمر شراء "
                "تلقائيًا، ولا يمثل ربحية الفئة.",
                "",
                f"في تحقق تاريخي على الأسابيع {int(validation_weeks[0])}–"
                f"{int(validation_weeks[-1])}، بلغ MAE للنموذج "
                f"**{rf_validation_mae:,.2f}** وحدة طلب لكل سجل. وعلى "
                f"{matched_coverage:.1f}% من سجلات التحقق ذات الأسبوع السابق "
                f"المرصود، بلغ MAE النموذج **{matched_rf_mae:,.2f}** مقابل "
                f"**{previous_week_mae:,.2f}** لخط أساس طلب الأسبوع السابق.",
                "",
                (
                    "خط أساس الأسبوع السابق كان أفضل في هذه المقارنة؛ لذلك "
                    "يُستخدم Random Forest كتقدير مساعد للمراجعة لا كبديل "
                    "تشغيلي لخط الأساس حتى يتحسن التحقق."
                    if matched_rf_mae > previous_week_mae
                    else "تفوق Random Forest على خط أساس الأسبوع السابق في "
                    "هذه المقارنة وحدها؛ يلزم تكرار التحقق الزمني قبل اعتماده."
                ),
                "",
                "**قيد مهم:** لا يحتوي `test.csv` على `num_orders` فعلية؛ لذلك "
                "لا يمكن حساب MAE أو تأكيد دقة تنبؤات أسابيع الاختبار. قيم MAE "
                "المذكورة تخص التحقق التاريخي فقط، وقد تختلف النتائج في أسابيع "
                "أو ظروف تشغيل أخرى.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"تم حفظ ملخص العميل: {insight_path}")

    # نعرض حجم الملفات الناتجة لتسهيل مراجعة التشغيل.
    print(f"عدد أسابيع test.csv: {test_data['week'].nunique():,}")
    print(
        "الذاكرة المستخدمة في train_data: "
        f"{train_data.memory_usage(deep=True).sum() / 1_000_000:.1f} MB"
    )


if __name__ == "__main__":
    main()
