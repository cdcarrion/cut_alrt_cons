# ============================================================
# Versión reducida de codigo_extraido.py
#
# Qué se quitó (todo lo demás se mantiene igual):
#  - Imports duplicados (pyspark.sql.functions*, Window, Bucketizer,
#    StorageLevel, NumericType, sklearn.tree/model_selection/metrics
#    aparecían repetidos varias veces a lo largo del script).
#  - Imports que no se usan en ninguna parte del código que se ve
#    aquí: SparkSession, VectorAssembler, ChiSqSelector,
#    VarianceThresholdSelector, RandomForestRegressor,
#    RandomForestClassifier, RegressionEvaluator,
#    MulticlassClassificationEvaluator, CrossValidator,
#    ParamGridBuilder, train_test_split, export_text, y el
#    `col` explícito (ya viene incluido en el wildcard de
#    pyspark.sql.functions).
#  - `import sys` y las dos líneas comentadas que dependían de él
#    (sys.path.append y el import de CustomClassificationMetrics).
#  - Los dos "# df_work_CE.unpersist()" comentados que quedaban
#    como recordatorio justo después del persist(): la llamada
#    real ya existe más abajo, así que el recordatorio sobraba.
#
# Nota: esta transcripción viene de fotos de pantalla, no del
# archivo original. Revisa sobre todo eda_continuas() y
# winsorize_df() (son las funciones más largas) por si algún
# número o nombre de columna quedó mal leído por el OCR.
# ============================================================

#!pip install optbinning
#!pip install pympler

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import NumericType
from pyspark import StorageLevel
from pyspark.ml.feature import Bucketizer

from optbinning import OptimalBinningSketch

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import tree


# ============================================================
# FUNCIONES OPTIMAL BINNING
# ============================================================

def add(partition):

    df_pandas = pd.DataFrame.from_records(
        partition,
        columns=columns
    )

    x = df_pandas[variable]
    y_local = df_pandas[target]

    optbsketch = OptimalBinningSketch(
        eps=1e-4,
        max_pvalue_policy="consecutive",
        monotonic_trend="auto_asc_desc",
        sketch="gk",
        min_bin_size=0.1,
        max_n_bins=5
    )

    optbsketch.add(x, y_local)

    return [optbsketch]


def merge(optbsketch, other_optbsketch):

    optbsketch.merge(other_optbsketch)

    return optbsketch


def winsorize_df(
    df: DataFrame,
    numeric_cols,
    lower: float = 0.02,
    upper: float = 0.98,
    approx_rel_error: float = 0.001,
    inplace: bool = True,
    suffix: str = "_win",
    verbose: bool = True,
):
    """
    Winsorize numeric columns in a Spark DataFrame by capping values below the `lower`
    percentile to that percentile and above the `upper` percentile to that percentile.
    - Keeps NULLs/NaNs as NULLs (no imputation).
    - Uses `approxQuantile` for performance.
    - Returns the transformed DataFrame and a dict of thresholds used.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input Spark DataFrame.
    numeric_cols : list[str]
        Columns to winsorize (must be numeric). Columns not found or not numeric will be skipped with a warning.
    lower : float, default 0.02
        Lower percentile (in [0,1)).
    upper : float, default 0.98
        Upper percentile (in (0,1]).
    approx_rel_error : float, default 0.001
        Relative error for `approxQuantile` (lower is more accurate but slower).
    inplace : bool, default True
        If True, overwrites the original columns; otherwise creates new columns with `suffix`.
    suffix : str, default "_win"
        Suffix for new columns if `inplace=False`.
    verbose : bool, default True
        Print progress and any columns skipped.

    Returns
    -------
    df_out : pyspark.sql.DataFrame
        DataFrame with winsorized columns.
    thresholds : dict[str, tuple(float, float)]
        Mapping column -> (lower_threshold, upper_threshold).
    """
    # ---- Validations ----
    if not 0 <= lower < upper <= 1:
        raise ValueError("Percentiles must satisfy: 0 <= lower < upper <= 1")

    numeric_cols = list(dict.fromkeys(numeric_cols or []))  # dedupe, tolerate None
    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Keep only numeric columns
    schema = df.schema
    def is_numeric(col):
        return isinstance(schema[col].dataType, NumericType)
    non_numeric = [c for c in numeric_cols if not is_numeric(c)]
    cols_to_process = [c for c in numeric_cols if is_numeric(c)]

    if verbose and non_numeric:
        print(f"[WARN] Skipping non-numeric columns: {non_numeric}")
    if not cols_to_process:
        if verbose:
            print("[INFO] No numeric columns to winsorize. Returning original DataFrame.")
        return df, {}

    # ---- Compute thresholds via approxQuantile ----
    thresholds = {}
    for c in cols_to_process:
        try:
            q_low, q_up = df.approxQuantile(c, [lower, upper], approx_rel_error)
            # If column has no non-null values, approxQuantile may return identical or NaN thresholds
            if q_low is None or q_up is None:
                if verbose:
                    print(f"[WARN] No valid values found for '{c}'. Skipping.")
                continue
            # Handle NaNs (rare but possible with all-NaN columns)
            if (q_low != q_low) or (q_up != q_up):  # NaN check
                if verbose:
                    print(f"[WARN] Quantiles NaN for '{c}'. Skipping.")
                continue
            thresholds[c] = (q_low, q_up)
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to compute quantiles for '{c}': {e}. Skipping.")
            continue

    if not thresholds:
        if verbose:
            print("[INFO] No thresholds computed. Returning original DataFrame.")
        return df, {}

    # ---- Apply capping (NULLs remain NULL) ----
    df_out = df
    for c, (q_low, q_up) in thresholds.items():
        # If for some reason lower > upper (degenerate), skip
        if q_low > q_up:
            if verbose:
                print(f"[WARN] lower quantile > upper quantile for '{c}'. Skipping.")
            continue

        new_col = c if inplace else f"{c}{suffix}"
        expr = (
            F.when(F.col(c) < F.lit(q_low), F.lit(q_low))
             .when(F.col(c) > F.lit(q_up), F.lit(q_up))
             .otherwise(F.col(c))
            # NOTE: if F.col(c) is NULL, comparisons are NULL, and .otherwise returns the original NULL
        )
        df_out = df_out.withColumn(new_col, expr)

    if verbose:
        print("[INFO] Winsorization complete. Columns processed:")
        for c, (q_low, q_up) in thresholds.items():
            tgt = c if inplace else f"{c}{suffix}"
            print(f"  - {c} -> {tgt} | 2%={q_low:.6g}, 98%={q_up:.6g}")

    return df_out, thresholds


def calculate_iv(data, var_list, label, var_mostrar=None):

    result = []
    for var in var_list:
        groupedTemp = data.groupBy(var, label).count()
        total_0 = groupedTemp.filter(col(label) == '0').agg(sum('count')).collect()[0][0]
        total_1 = groupedTemp.filter(col(label) == '1').agg(sum('count')).collect()[0][0]
        groupedData = (
            groupedTemp
            .filter(col(label) == '0')
            .withColumnRenamed('count', 'count_0')
            .alias('a')
            .join(
                groupedTemp
                .filter(col(label) == '1')
                .withColumnRenamed('count', 'count_1')
                .alias('b'),
                col('a.' + var) == col('b.' + var),
                'left'
            ).select(
                'a.*',
                'b.count_1'
            )
        )

        groupedData = (
            groupedData.withColumn(
                'WOE',
                log((col('count_0') / total_0) / (col('count_1') / total_1))
            )
            .withColumn(
                'IV',
                ((col('count_0') / total_0) - (col('count_1') / total_1)) * col('WOE')
            )
        )

        iv_temp = groupedData.agg(sum('IV')).collect()[0][0]
        result.append((var, iv_temp))

        if var_mostrar is not None:
            if var in var_mostrar:
                groupedData.show()

    IV = spark.createDataFrame(result, ['Variables', 'IV'])
    return IV


# EDA CONTINUAS
def eda_continuas(
    datos,
    variables,
    estadisticos=['count', 'mean', 'stddev', 'min',
                  '1%', '5%', '10%', '25%', '50%',
                  '75%', '90%', '95%', '99%', 'max'],
    bpnulos=50,
    bpatipicos=5,
):
    datos = datos.select(*variables)
    nulos = datos.select([count(when(col(var).isNull(), var)).alias(var) for var in datos.columns])
    nulos = nulos.select(lit("Nulos").alias("summary"), "*")
    resumen = datos.summary(estadisticos)
    resumen = nulos.union(resumen)
    resumen = resumen.selectExpr(
        "summary",
        f"""stack({len(variables)}, {','.join(map(','.join, zip([f"'{i}'" for i in variables], [f"{i}" for i in variables])))}) as ({','.join(["Variable", "Valor"])})"""
    )
    resumen = resumen.withColumn("Valor", col("Valor").cast("double"))
    resumen = resumen.groupBy("Variable").pivot("summary").sum("Valor")
    resumen = (
        resumen.withColumn("Registros", col("count") + col("Nulos"))
        .withColumn("PorcentajeNulos", round(100 * col("Nulos") / col("Registros"), 2))
        .withColumn("IQR", col("75%") - col("25%"))
        .withColumn("BandaInferior", col("25%") - 1.5 * col("IQR"))
        .withColumn("BandaSuperior", col("75%") + 1.5 * col("IQR"))
    )
    banda = resumen.select("Variable", "BandaInferior", "BandaSuperior")
    atipicos = []
    for i in range(banda.count()):
        a = banda.collect()[i]
        atipicos.append([a[0], datos.filter((col(a[0]) < a[1]) | (col(a[0]) > a[2])).count()])
    atipicos = spark.createDataFrame(atipicos, ["Variable", "Atipicos"])
    resumen = resumen.join(atipicos, how="left", on="Variable")
    resumen = resumen.withColumn(
        "PorcentajeAtipicos", round(100 * col("Atipicos") / col("Registros"), 2)
    ).withColumn(
        "Observaciones",
        when(
            (col("PorcentajeNulos") > bpnulos) & (col("PorcentajeAtipicos") > bpatipicos),
            "Porcentaje de Nulos mayor al " + str(bpnulos) + "% y Porcentaje de Atipicos mayor al " + str(bpatipicos) + "%",
        )
        .when(
            col("PorcentajeNulos") > bpnulos,
            "Porcentaje de Nulos mayor al " + str(bpnulos) + "%",
        )
        .when(
            col("PorcentajeAtipicos") > bpatipicos,
            "Porcentaje de Atipicos mayor al " + str(bpatipicos) + "%",
        ),
    )

    resumen = resumen.select(
        col("Variable").alias("variable"),
        col("Registros").alias("registros"),
        col("Nulos").alias("nulos"),
        col("PorcentajeNulos").alias("porcNulos"),
        *estadisticos,
        col("IQR").alias("iQR"),
        col("BandaInferior").alias("bandaInferior"),
        col("BandaSuperior").alias("bandaSuperior"),
        col("Atipicos").alias("atipicos"),
        col("PorcentajeAtipicos").alias("porcentajeAtipicos"),
        col("Observaciones").alias("observaciones"),
    )

    return resumen


# ============================================================
# CARGA Y PREPARACIÓN DE LA MDT
# ============================================================

path_mdt_final = 's3a://s3-lagodatos-noprod-06/data/pro/bp/tr_administracion_credito/modelos/admisionpersonas/10082/ZP_BP_Rie_TM_MdtPersonasDTree'

df_mdt_final = (
    spark.read.format('delta').load(path_mdt_final)
)

id_col = "identificacionCliente"
month_col = "codigoPeriodoConcesion"
y_col = "y"

w = Window.partitionBy(id_col, month_col).orderBy(col(y_col).desc())

df_modeling = (
    df_mdt_final
    .filter(
        ((col('numeroRegistrosRiesgo') >= 12) & (col('y') == '0')) | (col('y') == '1'))
    .filter(col('y').isNotNull())
    .withColumn("rn", row_number().over(w))
    .filter(col("rn") == 1)
    .drop("rn")
)

df_modeling = df_modeling.withColumn(
    'experiencia',
    when(col('reportesActivoU12m') >= 6, 'CE').otherwise('SE')
)

df_CE = df_modeling.filter('experiencia = "CE"')
df_SE = df_modeling.filter('experiencia = "SE"')

df_CE = df_CE.filter('codigoPeriodoConcesion >= 202401').withColumn(
    'particion', when(col('codigoPeriodoConcesion') < '202411', 'train').otherwise('test'))
df_SE = df_SE.filter('codigoPeriodoConcesion >= 202401').withColumn(
    'particion', when(col('codigoPeriodoConcesion') < '202411', 'train').otherwise('test'))

df_CE = df_CE.withColumn('maloEvidente', when(col('moraMesAnterior') > 90, 1).otherwise(0))
df_SE = df_SE.withColumn('maloEvidente', when(col('moraMesAnterior') > 90, 1).otherwise(0))

df_CE.groupBy('maloEvidente').count().show()
df_SE.groupBy('maloEvidente').count().show()

df_mdt_final.count()

df_mdt_final.toPandas().to_csv(
    "/var/sds/homes/crcarrio-bancopichincha/workspace/6_AlertaConsumo/df_mdt_final.csv",
    index=False
)

df_CE = df_CE.filter('maloEvidente = 0')
df_SE = df_SE.filter('maloEvidente = 0')

df_CE = df_CE.withColumn('weight', when(col('y') == 1, 16.22).otherwise(0.515))
df_SE = df_SE.withColumn('weight', when(col('y') == 1, 12.74).otherwise(0.520))

var_id = [
    'identificacionCliente',
    'codigoPeriodoConcesion',
    'y',
    'experiencia',
    'particion',
    'weight'
]
var = [
    'scoreBuro',
    'clasificacion',
    'antiguedadBuroSBS',
    'saldoCuentas'
]

wind_var = [
    'antiguedadBuroSBS',
    'saldoCuentas'
]

df_CE, thr = winsorize_df(
    df=df_CE,
    numeric_cols=wind_var,
    lower=0.02,
    upper=0.98,
    approx_rel_error=0.001,   # tighter = more accurate, slower
    inplace=True,             # overwrite the same columns
    verbose=True
)

# Inspect thresholds used:
print(thr)

# 1) Build your reusable base DataFrame (expensive steps here)
df_work_CE = df_CE
df_work_SE = df_SE

# 2) Persist the base if you will reuse it several times
df_work_CE = df_work_CE.persist(StorageLevel.MEMORY_AND_DISK)
df_work_SE = df_work_SE.persist(StorageLevel.MEMORY_AND_DISK)

df_CE = df_CE.withColumn('saldoCuentas', when(col('saldoCuentas') >= 9855.32, 9855.32).otherwise(col('saldoCuentas')))
df_CE = df_CE.withColumn('saldoCuentas', when(col('saldoCuentas') < 0, 0).otherwise(col('saldoCuentas')))
df_CE = df_CE.withColumn('antiguedadBuroSBS', when(col('antiguedadBuroSBS') >= 227, 227).otherwise(col('antiguedadBuroSBS')))
df_SE = df_SE.withColumn('saldoCuentas', when(col('saldoCuentas') >= 9163.58, 9163.58).otherwise(col('saldoCuentas')))
df_SE = df_SE.withColumn('saldoCuentas', when(col('saldoCuentas') < 0, 0).otherwise(col('saldoCuentas')))
df_SE = df_SE.withColumn('antiguedadBuroSBS', when(col('antiguedadBuroSBS') >= 219.0, 219.0).otherwise(col('antiguedadBuroSBS')))

dfEda = df_work_CE
variables = ['scoreBuro', 'antiguedadBuroSBS', 'saldoCuentas', 'y']
edaContinuas = eda_continuas(datos=dfEda, variables=variables)
edaContinuas.show(50, truncate=False)

dfEda = df_work_SE
variables = ['scoreBuro', 'antiguedadBuroSBS', 'saldoCuentas']
edaContinuas = eda_continuas(datos=dfEda, variables=variables)
edaContinuas.show(50, truncate=False)


# ============================================================
# OPTIMAL BINNING — df_work_CE
# ============================================================

var_modeling = ['antiguedadBuroSBS', 'saldoCuentas']
dataModel = df_work_CE

data_sample = dataModel
dict_df_variable = {}
dict_var_num, dict_var_cat, dict_var_dic = [], [], []
n_partitions = 4
model_df = data_sample.repartition(n_partitions)

target = "y"
dict_var_num = []

for variable in var_modeling:
    model_df = model_df.withColumn(variable, col(variable).cast("double"))

    columns = [variable, target]
    optbsketch = (
        model_df.select(columns)
        .rdd.mapPartitions(lambda partition: add(partition))
        .treeReduce(merge)
    )

    optbsketch.solve()
    tabla_iv = optbsketch.binning_table.build()
    bin_df = tabla_iv.loc[~tabla_iv["Bin"].isin(["Special"])]
    tabla_load = bin_df.drop(["Totals"], axis=0)
    tabla_load["variable"] = variable
    tabla_load["binsNumbers"] = range(tabla_load.shape[0])
    val_null = tabla_load[tabla_load["Bin"] == "Missing"].reset_index(drop=True)["binsNumbers"][0]

    if len(optbsketch.splits) > 1:
        var_bin = variable + "bins"
        model_df = Bucketizer(
            splits=list([-math.inf] + list(optbsketch.splits) + [math.inf]),
            inputCol=variable,
            outputCol=var_bin,
        ).transform(model_df)
        model_df = model_df.drop(variable)
        model_df = model_df.withColumn(variable, col(var_bin).cast("integer"))
        model_df = model_df.drop(var_bin)
        model_df = model_df.withColumn(
            variable,
            when(col(variable).isNull(), int(val_null)).otherwise(col(variable)),
        )

        dict_var_num.append(tabla_load)
        list_split = list(optbsketch.splits)
        list_split = [np.float(np.round(x, 2)) for x in list_split]
        dict_df_variable[variable] = list_split

total_iv = []
variable_name = []

for i, df_i in enumerate(dict_var_num, start=0):
    display(df_i.sort_values(by="Event rate", ascending=True, na_position="last"))
    iv_sum = df_i["IV"].sum(skipna=True)
    total_iv.append(iv_sum)
    variable_name.append(df_i["variable"].iat[0])

num_iv_df = pd.DataFrame({"variable": variable_name, "IV_sum": total_iv})
num_iv_df = num_iv_df.sort_values(by="IV_sum", ascending=False).reset_index(drop=True)
display(num_iv_df)

df_CE = df_CE.withColumn(
    'saldoCuentas_bin',
    when(col('saldoCuentas') > 552, 'c_0')
    .when(col('saldoCuentas') > 200, 'c_1')
    .when(col('saldoCuentas') > 63, 'c_2')
    .when(col('saldoCuentas') > 7.26, 'c_3')
    .when(col('saldoCuentas') <= 7.26, 'c_4')
    .otherwise('c_2')
).withColumn(
    'antiguedadBuroSBS_bin',
    when(col('antiguedadBuroSBS') > 140, 'c_0')
    .when(col('antiguedadBuroSBS') > 71, 'c_1')
    .when(col('antiguedadBuroSBS') > 41, 'c_2')
    .when(col('antiguedadBuroSBS') > 25, 'c_3')
    .when(col('antiguedadBuroSBS') <= 25, 'c_4')
    .otherwise('c_4')
)

df_SE = df_SE.withColumn(
    'saldoCuentas_bin',
    when(col('saldoCuentas') > 1028, 'c_0')
    .when(col('saldoCuentas') > 469, 'c_1')
    .when(col('saldoCuentas') > 92, 'c_2')
    .when(col('saldoCuentas') > 6.14, 'c_3')
    .when(col('saldoCuentas') <= 6.14, 'c_4')
    .otherwise('c_4')
).withColumn(
    'antiguedadBuroSBS_bin',
    when(col('antiguedadBuroSBS') > 128, 'c_0')
    .when(col('antiguedadBuroSBS') > 54, 'c_1')
    .when(col('antiguedadBuroSBS') > 20, 'c_2')
    .when(col('antiguedadBuroSBS') <= 20, 'c_4')
    .otherwise('c_3')
)

df_work_CE.unpersist()
df_work_SE.unpersist()


# ============================================================
# OPTIMAL BINNING — df_work_SE
# ============================================================

var_modeling = ['antiguedadBuroSBS', 'saldoCuentas']
dataModel = df_work_SE

data_sample = dataModel
dict_df_variable = {}
dict_var_num, dict_var_cat, dict_var_dic = [], [], []
n_partitions = 4
model_df = data_sample.repartition(n_partitions)

target = "y"
dict_var_num = []

for variable in var_modeling:
    model_df = model_df.withColumn(variable, col(variable).cast("double"))

    columns = [variable, target]
    optbsketch = (
        model_df.select(columns)
        .rdd.mapPartitions(lambda partition: add(partition))
        .treeReduce(merge)
    )

    optbsketch.solve()
    tabla_iv = optbsketch.binning_table.build()
    bin_df = tabla_iv.loc[~tabla_iv["Bin"].isin(["Special"])]
    tabla_load = bin_df.drop(["Totals"], axis=0)
    tabla_load["variable"] = variable
    tabla_load["binsNumbers"] = range(tabla_load.shape[0])
    val_null = tabla_load[tabla_load["Bin"] == "Missing"].reset_index(drop=True)["binsNumbers"][0]

    if len(optbsketch.splits) > 1:
        var_bin = variable + "bins"
        model_df = Bucketizer(
            splits=list([-math.inf] + list(optbsketch.splits) + [math.inf]),
            inputCol=variable,
            outputCol=var_bin,
        ).transform(model_df)
        model_df = model_df.drop(variable)
        model_df = model_df.withColumn(variable, col(var_bin).cast("integer"))
        model_df = model_df.drop(var_bin)
        model_df = model_df.withColumn(
            variable,
            when(col(variable).isNull(), int(val_null)).otherwise(col(variable)),
        )

        dict_var_num.append(tabla_load)
        list_split = list(optbsketch.splits)
        list_split = [np.float(np.round(x, 2)) for x in list_split]
        dict_df_variable[variable] = list_split

total_iv = []
variable_name = []

for i, df_i in enumerate(dict_var_num, start=0):
    display(df_i.sort_values(by="Event rate", ascending=True, na_position="last"))
    iv_sum = df_i["IV"].sum(skipna=True)
    total_iv.append(iv_sum)
    variable_name.append(df_i["variable"].iat[0])

num_iv_df = pd.DataFrame({"variable": variable_name, "IV_sum": total_iv})
num_iv_df = num_iv_df.sort_values(by="IV_sum", ascending=False).reset_index(drop=True)
display(num_iv_df)

var_bin = [
    'saldoCuentas_bin',
    'antiguedadBuroSBS_bin'
]

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

calculate_iv(df_work_CE, var_bin, label='y', var_mostrar=var_bin)


# ============================================================
# WOE — df_CE
# ============================================================

df_CE = df_CE.withColumn(
    'saldoCuentas_woe',
    when(col('saldoCuentas') > 552, 0.4944)
    .when(col('saldoCuentas') > 200, 0.2025)
    .when(col('saldoCuentas') > 63, -0.0476)
    .when(col('saldoCuentas') > 7.26, -0.2367)
    .when(col('saldoCuentas') <= 7.26, -0.6276)
    .otherwise(-0.0476)
).withColumn(
    'scoreBuro_woe',
    when(col('scoreBuro') < 800, -1.3359)
    .when(col('scoreBuro') < 894, -0.6230)
    .when(col('scoreBuro') < 920, -0.3504)
    .when(col('scoreBuro') < 960, -0.069)
    .when(col('scoreBuro') < 971, 0.3419)
    .when(col('scoreBuro') < 979, 0.54579)
    .when(col('scoreBuro') >= 979, 1.1462)
    .otherwise(-1.3359)
).withColumn(
    'clasificacion_woe',
    when(col('clasificacion').cast('float') == 1, 0.6764)
    .when(col('clasificacion').cast('float') == 2, 0.0757)
    .when(col('clasificacion').cast('float') == 3, -0.378)
    .otherwise(-0.93167)
).withColumn(
    'antiguedadBuroSBS_woe',
    when(col('antiguedadBuroSBS') > 140, 0.3883)
    .when(col('antiguedadBuroSBS') > 71, 0.18447)
    .when(col('antiguedadBuroSBS') > 41, -0.02339)
    .when(col('antiguedadBuroSBS') > 25, -0.0939)
    .when(col('antiguedadBuroSBS') <= 25, -0.3446)
    .otherwise(-0.3446)
)

woeVar_final = [
    'antiguedadBuroSBS_woe',
    'saldoCuentas_woe',
    'scoreBuro'
]

pd_train_CE = df_CE.toPandas()

pd_train_grid = pd_train_CE.sample(int(0.1 * len(pd_train_CE)))

X = pd_train_grid[woeVar_final]
y = pd_train_grid['y']
w = pd_train_grid['weight']

mask_train = pd_train_grid['particion'] == 'train'
mask_test = pd_train_grid['particion'] == 'test'

X_train = X.loc[mask_train].copy()
X_test = X.loc[mask_test].copy()
y_train = y.loc[mask_train].copy()
y_test = y.loc[mask_test].copy()
w_train = w.loc[mask_train].copy()

min_samples_leaf_5p = int(0.05 * len(X_train))

# ---- SIMPLE Grid Search (tiny search) — CE, 3 variables ----
param_grid = {
    "max_depth": [3, 4, 5],
}

model = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=min_samples_leaf_5p,
    random_state=13
)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train, sample_weight=w_train)

print("Best params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

best_model = grid.best_estimator_
best_model

importances = best_model.feature_importances_
features = X_train.columns
df_importance = pd.DataFrame({
    'variable': features,
    'importancia': importances
}).sort_values(by='importancia', ascending=False)
print(df_importance)

# ---- Árbol final CE, 2 variables (saldoCuentas_woe, scoreBuro sin WOE) ----
woeVar_final_CE = [
    'saldoCuentas_woe',
    'scoreBuro'
]

X = pd_train_CE[woeVar_final_CE]
y = pd_train_CE['y']
w = pd_train_CE['weight']

mask_train = pd_train_CE['particion'] == 'train'
mask_test = pd_train_CE['particion'] == 'test'

X_train = X.loc[mask_train].copy()
X_test = X.loc[mask_test].copy()
y_train = y.loc[mask_train].copy()
y_test = y.loc[mask_test].copy()
w_train = w.loc[mask_train].copy()

min_samples_leaf_5p = int(0.05 * len(X_train))

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=min_samples_leaf_5p,
    random_state=13
)

model.fit(X_train, y_train, sample_weight=w_train)

importances = model.feature_importances_
features = X_train.columns
df_importance = pd.DataFrame({
    'variable': features,
    'importancia': importances
}).sort_values(by='importancia', ascending=False)
print(df_importance)

print('-------------TEST----------')
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC del modelo Decision Tree: {auc:.4f}")
print(f"GINI del modelo Decision Tree: {2*auc-1:.4f}")

print('-------------TRAIN---------')
y_pred_proba = model.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred_proba)
print(f"AUC del modelo Decision Tree: {auc:.4f}")
print(f"GINI del modelo Decision Tree: {2*auc-1:.4f}")

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Punto de corte óptimo: {optimal_threshold:.4f}")

plt.figure(figsize=(15, 7))
tree.plot_tree(model, feature_names=X_train.columns, class_names=["Bueno", "Malo"], filled=True, proportion=True)
plt.title("Árbol de decisión")
plt.show()


# ============================================================
# WOE — df_SE
# ============================================================

df_SE = df_SE.withColumn(
    'saldoCuentas_woe',
    when(col('saldoCuentas') > 1028, 0.7714)
    .when(col('saldoCuentas') > 469, 0.5191)
    .when(col('saldoCuentas') > 92, 0.2153)
    .when(col('saldoCuentas') > 6.14, -0.1453)
    .when(col('saldoCuentas') <= 6.14, -0.79418)
    .otherwise(-0.79418)
).withColumn(
    'antiguedadBuroSBS_woe',
    when(col('antiguedadBuroSBS') > 128, 0.40036)
    .when(col('antiguedadBuroSBS') > 54, 0.1362)
    .when(col('antiguedadBuroSBS') > 20, 0.0798)
    .when(col('antiguedadBuroSBS') <= 20, -0.19288)
    .otherwise(-0.15734)
).withColumn(
    'scoreBuro_woe',
    when(col('scoreBuro') < 800, -0.3607)
    .when(col('scoreBuro') < 869, -0.4028)
    .when(col('scoreBuro') < 873, -0.22657)
    .when(col('scoreBuro') < 920, -0.1344)
    .when(col('scoreBuro') < 940, 0.06496)
    .when(col('scoreBuro') < 960, 0.4246)
    .when(col('scoreBuro') < 972, 0.532)
    .when(col('scoreBuro') >= 972, 1.20337)
    .otherwise(-0.3607)
).withColumn(
    'clasificacion_woe',
    when(col('clasificacion').cast('float') == 1.0, 0.876159)
    .when(col('clasificacion').cast('float') == 2, 0.32346)
    .when(col('clasificacion').cast('float') == 3, -0.2124)
    .when(col('clasificacion').cast('float') == 4, -0.48007)
    .when(col('clasificacion').cast('float') == 5, -0.2144)
    .when(col('clasificacion').cast('float') == 6, -0.48007)
    .otherwise(-0.2144)
)

pd_train_SE = df_SE.toPandas()

pd_train_grid = pd_train_SE.sample(int(0.1 * len(pd_train_SE)))

X = pd_train_grid[woeVar_final]
y = pd_train_grid['y']
w = pd_train_grid['weight']

mask_train = pd_train_grid['particion'] == 'train'
mask_test = pd_train_grid['particion'] == 'test'

X_train = X.loc[mask_train].copy()
X_test = X.loc[mask_test].copy()
y_train = y.loc[mask_train].copy()
y_test = y.loc[mask_test].copy()
w_train = w.loc[mask_train].copy()

min_samples_leaf_5p = int(0.05 * len(X_train))

# ---- SIMPLE Grid Search (tiny search) — SE, 3 variables ----
param_grid = {
    "max_depth": [3, 4, 5],
}

model = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=min_samples_leaf_5p,
    random_state=13
)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train, sample_weight=w_train)

print("Best params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

best_model = grid.best_estimator_
best_model

importances = best_model.feature_importances_
features = X_train.columns
df_importance = pd.DataFrame({
    'variable': features,
    'importancia': importances
}).sort_values(by='importancia', ascending=False)
print(df_importance)

# ---- Árbol SE, 3 variables (antiguedadBuroSBS_woe, saldoCuentas_woe, scoreBuro sin WOE) ----
woeVar_final_SE = [
    'antiguedadBuroSBS_woe',
    'saldoCuentas_woe',
    'scoreBuro'
]

X = pd_train_SE[woeVar_final_SE]
y = pd_train_SE['y']
w = pd_train_SE['weight']

mask_train = pd_train_SE['particion'] == 'train'
mask_test = pd_train_SE['particion'] == 'test'

X_train = X.loc[mask_train].copy()
X_test = X.loc[mask_test].copy()
y_train = y.loc[mask_train].copy()
y_test = y.loc[mask_test].copy()
w_train = w.loc[mask_train].copy()

min_samples_leaf_5p = int(0.05 * len(X_train))

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=min_samples_leaf_5p,
    random_state=13
)

model.fit(X_train, y_train, sample_weight=w_train)

importances = model.feature_importances_
features = X_train.columns
df_importance = pd.DataFrame({
    'variable': features,
    'importancia': importances
}).sort_values(by='importancia', ascending=False)
print(df_importance)

print('-------------TEST----------')
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC del modelo Decision Tree: {auc:.4f}")
print(f"GINI del modelo Decision Tree: {2*auc-1:.4f}")

print('-------------TRAIN---------')
y_pred_proba = model.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred_proba)
print(f"AUC del modelo Decision Tree: {auc:.4f}")
print(f"GINI del modelo Decision Tree: {2*auc-1:.4f}")

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Punto de corte óptimo: {optimal_threshold:.4f}")

plt.figure(figsize=(15, 7))
tree.plot_tree(model, feature_names=X_train.columns, class_names=["Bueno", "Malo"], filled=True, proportion=True)
plt.title("Árbol de decisión")
plt.show()

# ---- Árbol SE, 2 variables (saldoCuentas_woe, scoreBuro_woe) ----
pd_train_SE = df_SE.toPandas()

woeVar_final_SE = [
    'saldoCuentas_woe',
    'scoreBuro_woe',
]

X = pd_train_SE[woeVar_final_SE]
y = pd_train_SE['y']
w = pd_train_SE['weight']

mask_train = pd_train_SE['particion'] == 'train'
mask_test = pd_train_SE['particion'] == 'test'

X_train = X.loc[mask_train].copy()
X_test = X.loc[mask_test].copy()
y_train = y.loc[mask_train].copy()
y_test = y.loc[mask_test].copy()
w_train = w.loc[mask_train].copy()

min_samples_leaf_5p = int(0.05 * len(X_train))

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=min_samples_leaf_5p,
    random_state=13
)

model.fit(X_train, y_train, sample_weight=w_train)

importances = model.feature_importances_
features = X_train.columns
df_importance = pd.DataFrame({
    'variable': features,
    'importancia': importances
}).sort_values(by='importancia', ascending=False)
print(df_importance)

print('-------------TEST----------')
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC del modelo Decision Tree: {auc:.4f}")
print(f"GINI del modelo Decision Tree: {2*auc-1:.4f}")

print('-------------TRAIN---------')
y_pred_proba = model.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred_proba)
print(f"AUC del modelo Decision Tree: {auc:.4f}")
print(f"GINI del modelo Decision Tree: {2*auc-1:.4f}")

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Punto de corte óptimo: {optimal_threshold:.4f}")

plt.figure(figsize=(15, 7))
tree.plot_tree(model, feature_names=X_train.columns, class_names=["Bueno", "Malo"], filled=True, proportion=True)
plt.title("Árbol de decisión")
plt.show()

# ---- Árbol final CE, 2 variables (saldoCuentas_woe, scoreBuro_woe) ----
pd_train_CE = df_CE.toPandas()

woeVar_final_CE = [
    'saldoCuentas_woe',
    'scoreBuro_woe'
]

X = pd_train_CE[woeVar_final_CE]
y = pd_train_CE['y']
w = pd_train_CE['weight']

mask_train = pd_train_CE['particion'] == 'train'
mask_test = pd_train_CE['particion'] == 'test'

X_train = X.loc[mask_train].copy()
X_test = X.loc[mask_test].copy()
y_train = y.loc[mask_train].copy()
y_test = y.loc[mask_test].copy()
w_train = w.loc[mask_train].copy()

min_samples_leaf_5p = int(0.05 * len(X_train))

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=min_samples_leaf_5p,
    random_state=13
)

model.fit(X_train, y_train, sample_weight=w_train)

importances = model.feature_importances_
features = X_train.columns
df_importance = pd.DataFrame({
    'variable': features,
    'importancia': importances
}).sort_values(by='importancia', ascending=False)
print(df_importance)

print('-------------TEST----------')
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC del modelo Decision Tree: {auc:.4f}")
print(f"GINI del modelo Decision Tree: {2*auc-1:.4f}")

print('-------------TRAIN---------')
y_pred_proba = model.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred_proba)
print(f"AUC del modelo Decision Tree: {auc:.4f}")
print(f"GINI del modelo Decision Tree: {2*auc-1:.4f}")

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Punto de corte óptimo: {optimal_threshold:.4f}")

plt.figure(figsize=(15, 7))
tree.plot_tree(model, feature_names=X_train.columns, class_names=["Bueno", "Malo"], filled=True, proportion=True)
plt.title("Árbol de decisión")
plt.show()


# ============================================================
# OPTIMAL BINNING — avgSaldoCuentasU3M / avgSaldoCuentasU6M
#
# Mismo análisis que se hizo para antiguedadBuroSBS/saldoCuentas,
# pero sobre el promedio de saldo de los últimos 3 y 6 meses en
# vez del saldo puntual a la fecha de corte. Corre este bloque
# primero para obtener las tablas de bins/WOE (dict_var_num,
# num_iv_df) de cada variable; con esos resultados en mano,
# agrega más abajo los withColumn(..._woe, ...) con los cortes
# reales — igual que se hizo con saldoCuentas_woe y scoreBuro_woe
# — y luego arma el árbol.
# ============================================================

dfEda = df_work_CE
variables = ['avgSaldoCuentasU3M', 'avgSaldoCuentasU6M', 'y']
edaContinuas = eda_continuas(datos=dfEda, variables=variables)
edaContinuas.show(50, truncate=False)

dfEda = df_work_SE
variables = ['avgSaldoCuentasU3M', 'avgSaldoCuentasU6M']
edaContinuas = eda_continuas(datos=dfEda, variables=variables)
edaContinuas.show(50, truncate=False)


# ---- Optimal binning — df_work_CE ----

var_modeling = ['avgSaldoCuentasU3M', 'avgSaldoCuentasU6M']
dataModel = df_work_CE

data_sample = dataModel
dict_df_variable = {}
dict_var_num, dict_var_cat, dict_var_dic = [], [], []
n_partitions = 4
model_df = data_sample.repartition(n_partitions)

target = "y"
dict_var_num = []

for variable in var_modeling:
    model_df = model_df.withColumn(variable, col(variable).cast("double"))

    columns = [variable, target]
    optbsketch = (
        model_df.select(columns)
        .rdd.mapPartitions(lambda partition: add(partition))
        .treeReduce(merge)
    )

    optbsketch.solve()
    tabla_iv = optbsketch.binning_table.build()
    bin_df = tabla_iv.loc[~tabla_iv["Bin"].isin(["Special"])]
    tabla_load = bin_df.drop(["Totals"], axis=0)
    tabla_load["variable"] = variable
    tabla_load["binsNumbers"] = range(tabla_load.shape[0])
    val_null = tabla_load[tabla_load["Bin"] == "Missing"].reset_index(drop=True)["binsNumbers"][0]

    if len(optbsketch.splits) > 1:
        var_bin = variable + "bins"
        model_df = Bucketizer(
            splits=list([-math.inf] + list(optbsketch.splits) + [math.inf]),
            inputCol=variable,
            outputCol=var_bin,
        ).transform(model_df)
        model_df = model_df.drop(variable)
        model_df = model_df.withColumn(variable, col(var_bin).cast("integer"))
        model_df = model_df.drop(var_bin)
        model_df = model_df.withColumn(
            variable,
            when(col(variable).isNull(), int(val_null)).otherwise(col(variable)),
        )

        dict_var_num.append(tabla_load)
        list_split = list(optbsketch.splits)
        list_split = [np.float(np.round(x, 2)) for x in list_split]
        dict_df_variable[variable] = list_split

total_iv = []
variable_name = []

for i, df_i in enumerate(dict_var_num, start=0):
    display(df_i.sort_values(by="Event rate", ascending=True, na_position="last"))
    iv_sum = df_i["IV"].sum(skipna=True)
    total_iv.append(iv_sum)
    variable_name.append(df_i["variable"].iat[0])

num_iv_df = pd.DataFrame({"variable": variable_name, "IV_sum": total_iv})
num_iv_df = num_iv_df.sort_values(by="IV_sum", ascending=False).reset_index(drop=True)
display(num_iv_df)


# ---- Optimal binning — df_work_SE ----

var_modeling = ['avgSaldoCuentasU3M', 'avgSaldoCuentasU6M']
dataModel = df_work_SE

data_sample = dataModel
dict_df_variable = {}
dict_var_num, dict_var_cat, dict_var_dic = [], [], []
n_partitions = 4
model_df = data_sample.repartition(n_partitions)

target = "y"
dict_var_num = []

for variable in var_modeling:
    model_df = model_df.withColumn(variable, col(variable).cast("double"))

    columns = [variable, target]
    optbsketch = (
        model_df.select(columns)
        .rdd.mapPartitions(lambda partition: add(partition))
        .treeReduce(merge)
    )

    optbsketch.solve()
    tabla_iv = optbsketch.binning_table.build()
    bin_df = tabla_iv.loc[~tabla_iv["Bin"].isin(["Special"])]
    tabla_load = bin_df.drop(["Totals"], axis=0)
    tabla_load["variable"] = variable
    tabla_load["binsNumbers"] = range(tabla_load.shape[0])
    val_null = tabla_load[tabla_load["Bin"] == "Missing"].reset_index(drop=True)["binsNumbers"][0]

    if len(optbsketch.splits) > 1:
        var_bin = variable + "bins"
        model_df = Bucketizer(
            splits=list([-math.inf] + list(optbsketch.splits) + [math.inf]),
            inputCol=variable,
            outputCol=var_bin,
        ).transform(model_df)
        model_df = model_df.drop(variable)
        model_df = model_df.withColumn(variable, col(var_bin).cast("integer"))
        model_df = model_df.drop(var_bin)
        model_df = model_df.withColumn(
            variable,
            when(col(variable).isNull(), int(val_null)).otherwise(col(variable)),
        )

        dict_var_num.append(tabla_load)
        list_split = list(optbsketch.splits)
        list_split = [np.float(np.round(x, 2)) for x in list_split]
        dict_df_variable[variable] = list_split

total_iv = []
variable_name = []

for i, df_i in enumerate(dict_var_num, start=0):
    display(df_i.sort_values(by="Event rate", ascending=True, na_position="last"))
    iv_sum = df_i["IV"].sum(skipna=True)
    total_iv.append(iv_sum)
    variable_name.append(df_i["variable"].iat[0])

num_iv_df = pd.DataFrame({"variable": variable_name, "IV_sum": total_iv})
num_iv_df = num_iv_df.sort_values(by="IV_sum", ascending=False).reset_index(drop=True)
display(num_iv_df)


# ============================================================
# TODO — una vez tengas las tablas de arriba (bins + WOE de
# avgSaldoCuentasU3M y avgSaldoCuentasU6M para CE y para SE),
# pega aquí los withColumn(..._woe, ...) con los cortes reales,
# por ejemplo:
#
# df_CE = df_CE.withColumn(
#     'avgSaldoCuentasU3M_woe',
#     when(col('avgSaldoCuentasU3M') > ___, ___)
#     ...
#     .otherwise(___)
# ).withColumn(
#     'avgSaldoCuentasU6M_woe',
#     when(col('avgSaldoCuentasU6M') > ___, ___)
#     ...
#     .otherwise(___)
# )
#
# (mismo patrón para df_SE) y después arma el árbol igual que
# antes, reemplazando 'saldoCuentas_woe' por 'avgSaldoCuentasU3M_woe'
# o 'avgSaldoCuentasU6M_woe' en woeVar_final_CE / woeVar_final_SE.
# ============================================================
