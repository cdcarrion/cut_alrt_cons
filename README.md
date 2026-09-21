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
