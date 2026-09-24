def transformar_woe(valor, cortes, woes, woe_missing):
    """
    cortes: lista de cortes ascendente, ej [37.69, 153.48, 287.13, 657.85]
    woes:   WOE de cada bin, de menor a mayor (len(woes) == len(cortes) + 1)
    """
    if valor is None:
        return woe_missing
    for corte, woe in zip(cortes, woes):
        if valor <= corte:
            return woe
    return woes[-1]  # cayó en el último bin (mayor al último corte)




Con Con Experiencia ya completé lo mismo: agregué avgSaldoCuentasU3M_woe y avgSaldoCuentasU6M_woe a la cadena withColumn de df_CE, usando los cortes exactos de tus tablas. Con esto ya tienes las 4 variables (CE y SE, U3M y U6M) listas para armar el árbol.
Sobre la interpretación:
Event rate (por bin): proporción de "malos" dentro de ese bin (Event / Count). Es la tasa de mora observada en ese segmento. Ej: en scoreBuro CE, el bin [979, inf) tiene event rate 0.0097 (casi nadie malo), y el bin (-inf, 877) tiene 0.0642 (6.5 veces más riesgo).
WoE: mide qué tan distinto es ese bin respecto al promedio general, en escala logarítmica de odds. WoE positivo = bin más seguro (menos malos de lo esperado), WoE negativo = bin más riesgoso. Por eso siempre vas a ver que WoE alto correlaciona con event rate bajo (como el ejemplo de arriba: WoE +1.177 ↔ event rate 0.97%).
IV por bin: cuánto aporta ESE bin específico al poder predictivo total de la variable. Depende de qué tan lejos está su WoE de cero Y qué tanta población tiene ese bin (un bin raro con WoE extremo aporta poco si casi no tiene clientes).
IV_sum: la suma de todos los IV de los bins = poder predictivo total de la variable. Regla general de referencia:
< 0.02 → sin poder predictivo
0.02 – 0.1 → débil
0.1 – 0.3 → medio
0.3 – 0.5 → fuerte
> 0.5 → sospechoso (posible fuga de información)
Con tus valores de CE: scoreBuro (0.281) y saldoCuentas (0.165) están en rango medio-fuerte; avgSaldoCuentasU3M (0.122) es medio pero notablemente menor a saldoCuentas; antiguedadBuroSBS (0.072) y avgSaldoCuentasU6M (0.071) son débiles.
Dato a tener en cuenta: en CE, saldoCuentas (punto de corte) sigue teniendo más IV que avgSaldoCuentasU3M (el promedio) — a diferencia de lo que buscabas al crear estas variables. Vale la pena comparar esto también contra los resultados de SE que ya tenías, para decidir si conviene reemplazar saldoCuentas por el promedio o usarlas juntas en el árbol.




REGLAS_FA_CE = [
    (979, None, 63, None, False, "C_11"),
    (960, 979, 552, None, False, "C_10"),
    (960, 979, 63, 552, False, "C_9"),
    (960, None, 7.26, 63, False, "C_8"),
    (894, 960, (552, False), None, False, "C_7"),   # saldo > 552 (estricto)
    (960, None, None, (7.26, True), False, "C_6"),  # saldo <= 7.26 (inclusive)
    (894, 960, 63, 552, False, "C_5"),               # ORIGINAL: saldo < 552 (sin corregir)
    (920, 960, 7.26, 63, False, "C_4"),
    (None, 894, 63, None, False, "C_3"),
    (920, 960, None, 7.26, False, "C_2"),
    (None, 920, None, 63, False, "C_1"),
]

REGLAS_FA_SE = [
    (972, None, 92, None, False, "C_12"),
    (940, 972, 469, None, False, "C_11"),
    (940, 972, 92, 469, False, "C_10"),
    (None, 940, 1028, None, False, "C_9"),
    (940, None, 6.14, 92, False, "C_8"),
    (None, 940, 469, 1028, False, "C_7"),
    (869, 940, 92, 469, False, "C_6"),
    (940, None, None, 6.14, False, "C_5"),
    (None, 869, 92, 469, False, "C_4"),
    (873, 940, 6.14, 92, False, "C_3"),
    (None, 873, 6.14, 92, False, "C_2"),
    (None, 920, None, 6.14, False, "C_1"),   # ORIGINAL: score < 920 (sin corregir)
]
