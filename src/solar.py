"""Preliminary rooftop-solar scenarios with explicit pricing assumptions."""


def solar_scenario(
    annual_kwh,
    capacity_kw,
    yield_kwh_per_kw=1400,
    cost_3kw=80000,
    cost_above_3kw=50000,
    subsidy_3kw=78000,
    tariff=7.0,
    self_consumption=0.8,
    maintenance_pct=0.01,
    daytime_consumption_share=0.45,
    export_rate=3.0,
    pricing_scenario="incremental_above_3kw",
):
    """Return a labelled scenario; verify policy and site inputs before use.

    The supplied project assumptions do not define costs below 3 kW. The
    dashboard therefore starts at 3 kW and exposes two alternative meanings
    for the above-3-kW rate instead of silently extrapolating a smaller system.
    """
    if annual_kwh < 0 or capacity_kw < 3:
        raise ValueError("annual_kwh must be non-negative and capacity_kw must be at least 3 kW")
    if pricing_scenario not in {"incremental_above_3kw", "all_capacity_above_3kw_rate"}:
        raise ValueError("pricing_scenario is not supported")

    generation = capacity_kw * yield_kwh_per_kw
    daytime_demand = annual_kwh * daytime_consumption_share
    onsite = min(generation * self_consumption, daytime_demand)
    export = max(generation - onsite, 0)

    if capacity_kw == 3:
        gross_cost = cost_3kw
    elif pricing_scenario == "incremental_above_3kw":
        gross_cost = cost_3kw + (capacity_kw - 3) * cost_above_3kw
    else:
        gross_cost = capacity_kw * cost_above_3kw

    subsidy = min(subsidy_3kw, gross_cost)
    net_cost = gross_cost - subsidy
    gross_savings = onsite * tariff + export * export_rate
    maintenance = gross_cost * maintenance_pct
    annual_net_savings = max(gross_savings - maintenance, 0)
    payback = net_cost / annual_net_savings if annual_net_savings else None
    return {
        "annual_consumption_kwh": annual_kwh,
        "capacity_kw": capacity_kw,
        "generation": generation,
        "on_site_consumption": onsite,
        "export": export,
        "gross_cost": gross_cost,
        "subsidy": subsidy,
        "net_cost": net_cost,
        "gross_savings": gross_savings,
        "maintenance": maintenance,
        "annual_net_savings": annual_net_savings,
        "payback": payback,
        "pricing_scenario": pricing_scenario,
        "status": "preliminary consumption-based scenario; verify technical and policy inputs",
    }
