#!/usr/bin/env python3
"""
Diagnostica ICON-2I RUC per previsioni.py.

Da lanciare in locale, nella stessa cartella di previsioni.py:

    pip install numpy requests eccodes
    python diagnostica_icon2i.py

Cosa fa:
  1. elenca i run disponibili e verifica quali sono completi (e con che ritardo);
  2. elenca tutte le cartelle/file del run con le dimensioni;
  3. scarica e analizza i GRIB delle variabili usate (e alcune extra):
     chiavi GRIB, griglia, unita', passi temporali, valori sul punto;
  4. confronta i modi di ricavare l'umidita' (Magnus da TD_2M, da QV_2M+PS, RELHUM al livello piu' basso);
  5. controlla TOT_PREC (cumulata? resettata? include neve?), vento (rotazione griglia), raffiche, nuvolosita';
  6. esegue previsioni.py end-to-end in una cartella temporanea e valida il JSON prodotto
     (+ secondo lancio per provare il ramo "run gia' completo").

Alla fine scrive un unico file  report_icon2i_<run>_<data>.md  da incollare/caricare in chat.

Opzioni:  --run AAAAMMGGHH   usa un run preciso      --skip-e2e   salta il test end-to-end
          --no-extras        salta le variabili extra  --max-3d N   messaggi da leggere da RELHUM (def. 150)
"""
import argparse
import contextlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import requests
import eccodes as ec

import previsioni as P

NEEDED = list(P.FOLDERS)
EXTRAS = ["RELHUM", "QV_2M", "PS", "RAIN_CON", "RAIN_GSP", "SNOW_CON", "SNOW_GSP", "PMSL", "WW"]
THREE_D = {"RELHUM"}

KEYS = [
    "shortName", "name", "units", "paramId", "discipline", "parameterCategory", "parameterNumber",
    "typeOfLevel", "level", "stepType", "stepUnits", "stepRange", "step", "forecastTime",
    "dataDate", "dataTime", "validityDate", "validityTime",
    "productDefinitionTemplateNumber", "typeOfStatisticalProcessing", "lengthOfTimeRange",
    "centre", "subCentre", "generatingProcessIdentifier", "tablesVersion", "localTablesVersion",
    "gridType", "Ni", "Nj", "numberOfPoints", "numberOfDataPoints", "numberOfMissing",
    "bitsPerValue", "packingType", "missingValue", "bitMapIndicator",
]
GRID_KEYS = [
    "gridType", "Ni", "Nj", "numberOfPoints",
    "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
    "latitudeOfLastGridPointInDegrees", "longitudeOfLastGridPointInDegrees",
    "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees",
    "iScansNegatively", "jScansPositively", "jPointsAreConsecutive",
    "latitudeOfSouthernPoleInDegrees", "longitudeOfSouthernPoleInDegrees", "angleOfRotationInDegrees",
    "southPoleLatitudeInDegrees", "southPoleLongitudeInDegrees",
    "uvRelativeToGrid", "resolutionAndComponentFlags", "numberOfGridUsed", "uuidOfHGrid",
]

# ----------------------------------------------------------------------------- report


class Report:
    def __init__(self):
        self.body = []
        self.checks = []

    def h(self, t, lvl=2):
        self.body += ["", "#" * lvl + " " + t, ""]

    def p(self, *a):
        self.body.append(" ".join(str(x) for x in a))

    def code(self, text):
        self.body += ["```", str(text).rstrip(), "```"]

    def table(self, header, rows):
        self.body.append("| " + " | ".join(map(str, header)) + " |")
        self.body.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self.body.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")

    def check(self, status, name, detail=""):
        self.checks.append((status, name, detail))
        print(f"  [{status}] {name} {detail}")

    def error(self, where):
        tb = traceback.format_exc()
        self.check("FAIL", f"eccezione in {where}", tb.strip().splitlines()[-1])
        self.code(tb)

    def render(self, title):
        head = [f"# {title}", ""]
        head += ["## Riepilogo controlli", ""]
        order = {"FAIL": 0, "WARN": 1, "INFO": 2, "OK": 3}
        for st, n, d in sorted(self.checks, key=lambda c: order.get(c[0], 9)):
            head.append(f"- **{st}** {n}" + (f" - {d}" if d else ""))
        return "\n".join(head + self.body) + "\n"


def clean_log(text):
    out = []
    for line in text.split("\n"):
        line = line.split("\r")[-1].rstrip()
        if line and not re.fullmatch(r"\s*\d+ MB", line):
            out.append(line)
    return "\n".join(out)


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return "nan" if math.isnan(v) else f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3g}"
    return v


def valid_time(h):
    return datetime.strptime(f"{P.g(h, 'validityDate')}{int(P.g(h, 'validityTime')):04d}",
                             "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


# ----------------------------------------------------------------------------- 1. run


def section_env(rep):
    rep.h("Ambiente")
    try:
        ecv = ec.codes_get_api_version()
    except Exception:
        ecv = "?"
    rep.table(["chiave", "valore"], [
        ["data/ora UTC", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")],
        ["python", sys.version.split()[0]], ["os", platform.platform()],
        ["eccodes (api)", ecv], ["numpy", np.__version__], ["requests", requests.__version__],
        ["ROOT", P.ROOT], ["punto", f"{P.LAT}, {P.LON}"], ["raggio", f"{P.RADIUS_KM} km"],
    ])


def section_runs(rep, forced):
    rep.h("Run disponibili")
    html = requests.get(P.ROOT, headers=P.HEADERS, timeout=60).text
    stamp = {m.group(1): m.group(2) for m in
             re.finditer(r'href="(\d{10})/"[^\n]*?(\d{2}-\w{3}-\d{4} \d{2}:\d{2})', html)}
    runs = sorted(stamp, reverse=True)
    rep.p(f"{len(runs)} run nel listing; il piu' recente e' {runs[0] if runs else '-'}.")
    rows, chosen = [], None
    for run in runs[:8]:
        miss, counts = [], []
        for f in NEEDED:
            try:
                n = len(P.data_files(f"{P.ROOT}{run}/{f}/"))
            except Exception:
                n = 0
            counts.append(n)
            if n == 0:
                miss.append(f)
        ok = not miss
        if ok and chosen is None:
            chosen = run
        rows.append([run, stamp[run], "completo" if ok else "mancano: " + ",".join(miss),
                     "/".join(map(str, counts))])
    rep.table(["run", "ultima modifica (listing)", "stato", f"file per cartella ({'/'.join(NEEDED)})"], rows)
    if forced:
        chosen = forced
        rep.check("INFO", "run forzato", forced)
    if not chosen:
        rep.check("FAIL", "nessun run completo trovato")
        return None
    rep.check("OK", "run completo scelto", chosen)
    try:
        t = datetime.strptime(stamp[chosen], "%d-%b-%Y %H:%M")
        rt = datetime.strptime(chosen, "%Y%m%d%H")
        rep.check("INFO", "ritardo pubblicazione ~", f"{(t - rt).total_seconds() / 3600:.1f} h dopo l'ora del run (ora del server, non garantita UTC)")
    except Exception:
        pass
    return chosen


def section_inventory(rep, run):
    rep.h("Inventario cartelle del run")
    base = f"{P.ROOT}{run}/"
    folders = [u.rstrip("/").split("/")[-1] for u in P.links(base) if u.endswith("/")]
    rows = []
    for f in folders:
        try:
            files = P.data_files(f"{base}{f}/")
            allf = [x for x in P.links(f"{base}{f}/") if not x.endswith("/")]
        except Exception as e:
            rows.append([f, "errore", str(e)[:60], ""])
            continue
        sizes = []
        for u in files[:3]:
            try:
                r = requests.head(u, headers=P.HEADERS, timeout=30, allow_redirects=True)
                sizes.append(f"{int(r.headers['Content-Length']) / 1e6:.1f} MB")
            except Exception:
                sizes.append("?")
        rows.append([f, len(files), ", ".join(x.split("/")[-1] for x in files[:3]) +
                     (" ..." if len(files) > 3 else ""), ", ".join(sizes),
                     ",".join(sorted({os.path.splitext(x)[1] for x in allf}))])
    rep.table(["cartella", "n file dati", "nomi (primi 3)", "dimensioni", "estensioni"], rows)
    return base


# ----------------------------------------------------------------------------- 2. GRIB


def describe(h):
    return {k: P.g(h, k) for k in KEYS}


def analyze_folder(rep, base, folder, max_msgs=None, max_files=2):
    """Scarica la cartella e restituisce i record (uno per messaggio)."""
    rep.h(f"{folder}", 3)
    try:
        files = P.data_files(f"{base}{folder}/")
    except Exception:
        rep.error(f"listing {folder}")
        return []
    if not files:
        rep.check("FAIL", f"{folder}: nessun file dati")
        return []
    if len(files) > 1:
        rep.check("INFO", f"{folder}: {len(files)} file", ", ".join(x.split("/")[-1] for x in files[:5]))
    records = []
    for url in files[:max_files]:
        name = url.split("/")[-1]
        t0, nbytes, n, idx = time.time(), 0, 0, None
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                for raw in P.iter_messages(url):
                    nbytes += len(raw)
                    n += 1
                    h = ec.codes_new_from_message(raw)
                    if idx is None:
                        idx, d, lats, lons = P.nearest(h)
                        near = np.where(d <= P.RADIUS_KM)[0]
                        grid = {k: P.g(h, k) for k in GRID_KEYS}
                        grid.update(dist_km=round(float(d[idx]), 2), n_near=len(near),
                                    lat_range=[float(lats.min()), float(lats.max())],
                                    lon_range=[float(lons.min()), float(lons.max())],
                                    cell=[float(lats[idx]), float(lons[idx])])
                    rec = describe(h)
                    vals = ec.codes_get_values(h)
                    miss = P.g(h, "missingValue")
                    ok = np.isfinite(vals) & ((vals != miss) if miss is not None else True)
                    rec.update(_valid=valid_time(h), _point=float(vals[idx]),
                               _min=float(vals[ok].min()) if ok.any() else None,
                               _max=float(vals[ok].max()) if ok.any() else None,
                               _mean=float(vals[ok].mean()) if ok.any() else None,
                               _nbad=int((~ok).sum()), _file=name, _grid=grid)
                    records.append(rec)
                    ec.codes_release(h)
                    if max_msgs and n >= max_msgs:
                        break
        except Exception:
            rep.error(f"download/lettura {name}")
        dt = time.time() - t0
        rep.p(f"**{name}**: {n} messaggi, {nbytes / 1e6:.1f} MB letti in {dt:.1f} s "
              f"({nbytes / 1e6 / max(dt, 1e-6):.1f} MB/s)" + (f" [limitato a {max_msgs} messaggi]" if max_msgs else ""))
    if not records:
        rep.check("FAIL", f"{folder}: nessun messaggio GRIB letto")
        return records

    # chiavi costanti / variabili
    const, var = {}, {}
    for k in KEYS:
        vals = [r[k] for r in records]
        u = list(dict.fromkeys(vals))
        (const if len(u) == 1 else var)[k] = u[0] if len(u) == 1 else u
    rep.p("")
    rep.p("Chiavi costanti:")
    rep.code("\n".join(f"{k} = {v}" for k, v in const.items()))
    rep.p("Chiavi variabili (valori distinti, max 12):")
    rep.code("\n".join(f"{k}: {v[:12]}{' ...' if len(v) > 12 else ''} ({len(v)} distinti)" for k, v in var.items())
             or "(nessuna)")
    g_now = {k: v for k, v in records[0]["_grid"].items() if v is not None}
    if analyze_folder.ref_grid is None:
        analyze_folder.ref_grid = g_now
        rep.p("Griglia / cella piu' vicina (uguale per le altre variabili salvo diversa indicazione):")
        rep.code(json.dumps(g_now, indent=1, default=str))
    elif g_now == analyze_folder.ref_grid:
        rep.p("Griglia: identica alla prima variabile.")
    else:
        diff = {k: v for k, v in g_now.items() if analyze_folder.ref_grid.get(k) != v}
        rep.check("WARN", f"{folder}: griglia diversa dalla prima variabile", str(list(diff)))
        rep.code(json.dumps(diff, indent=1, default=str))

    # tabella messaggi
    rows = []
    for r in records:
        rows.append([r["_valid"].strftime("%d %H:%M"), r["stepRange"], r["level"],
                     fmt(r["_point"]), fmt(r["_min"]), fmt(r["_mean"]), fmt(r["_max"]), r["_nbad"]])
    show = rows if len(rows) <= 60 else rows[:12] + [["...", "", "", "", "", "", "", ""]] + rows[-6:]
    rep.table(["valido (UTC)", "stepRange", "livello", "valore punto", "min", "media", "max", "mancanti"], show)

    # controlli generali
    if max_msgs is None:
        times = sorted({r["_valid"] for r in records if r["level"] in (records[0]["level"],)})
        if len(times) > 1:
            steps = sorted({int((b - a).total_seconds() // 60) for a, b in zip(times, times[1:])})
            rep.check("INFO", f"{folder}: passo temporale (min)", f"{steps}, {len(times)} istanti, "
                      f"{times[0]:%d %H:%M} -> {times[-1]:%d %H:%M}")
            if steps != [60]:
                rep.check("WARN", f"{folder}: passo non orario", f"{steps}")
    bad = sum(r["_nbad"] for r in records)
    if bad:
        rep.check("WARN", f"{folder}: valori mancanti/NaN nel dominio", str(bad))
    return records


analyze_folder.ref_grid = None


def series(records, level=None):
    out = {}
    for r in records:
        if level is not None and r["level"] != level:
            continue
        out[r["_valid"]] = r["_point"]
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------- 3. controlli fisici


def magnus_rh(t_k, td_k):
    return P.rh_from_dewpoint(t_k, td_k)


def rh_from_q(t_k, q, p_pa):
    e = q * p_pa / (0.622 + 0.378 * q)
    tc = t_k - 273.15
    es = 611.2 * math.exp(17.62 * tc / (243.12 + tc))
    return min(100.0, 100 * e / es)


def rotation_angle(grid, lat, lon):
    """Angolo (gradi) tra i assi della griglia ruotata e meridiani/paralleli nel punto (solo modulo)."""
    sp_lat = grid.get("latitudeOfSouthernPoleInDegrees")
    sp_lon = grid.get("longitudeOfSouthernPoleInDegrees")
    if sp_lat is None or sp_lon is None:
        return None
    polat, polon = math.radians(-sp_lat), math.radians(sp_lon + 180)
    la, lo = math.radians(lat), math.radians(lon)
    zsin = math.cos(polat) * math.sin(polon - lo)
    zcos = math.cos(la) * math.sin(polat) - math.sin(la) * math.cos(polat) * math.cos(polon - lo)
    return math.degrees(math.atan2(zsin, zcos))


def section_checks(rep, rec):
    rep.h("Controlli fisici sul punto")
    S = {k: series(v) for k, v in rec.items() if v and k not in THREE_D}
    g0 = next((r[0]["_grid"] for r in rec.values() if r), None)

    # --- temperatura
    if "T_2M" in S:
        v = list(S["T_2M"].values())
        rep.check("OK" if 220 < min(v) and max(v) < 330 else "FAIL", "T_2M in kelvin plausibile",
                  f"{min(v):.1f}..{max(v):.1f} K = {min(v) - 273.15:.1f}..{max(v) - 273.15:.1f} C")

    # --- umidita'
    rep.h("Umidita' relativa: confronto metodi", 3)
    T, TD, Q, PS = S.get("T_2M"), S.get("TD_2M"), S.get("QV_2M"), S.get("PS")
    rows = []
    if T and TD:
        diffq = []
        for t in T:
            if t not in TD:
                continue
            a = magnus_rh(T[t], TD[t])
            b = rh_from_q(T[t], Q[t], PS[t]) if Q and PS and t in Q and t in PS else None
            if b is not None:
                diffq.append(a - b)
            rows.append([t.strftime("%d %H:%M"), fmt(T[t] - 273.15, 1), fmt(TD[t] - 273.15, 1),
                         fmt(a, 1), fmt(b, 1) if b is not None else "-"])
        rep.table(["valido", "T2m C", "Td2m C", "UR Magnus(T,Td) %", "UR da QV_2M+PS %"], rows[:30])
        if diffq:
            rep.check("OK" if np.mean(np.abs(diffq)) < 2 else "WARN", "Magnus(T,Td) vs QV_2M+PS",
                      f"scarto medio assoluto {np.mean(np.abs(diffq)):.2f} punti %")
        if any(TD[t] > T[t] + 0.01 for t in T if t in TD):
            rep.check("WARN", "TD_2M > T_2M in alcuni istanti (UR clippata a 100)")
        elif not rows:
            rep.check("WARN", "T_2M e TD_2M senza istanti in comune")
        else:
            rep.check("OK", "TD_2M <= T_2M sempre")
    else:
        rep.check("FAIL", "mancano T_2M/TD_2M per calcolare l'umidita'")

    rh3 = rec.get("RELHUM")
    if rh3:
        levels = sorted({r["level"] for r in rh3 if r["level"] is not None})
        t0 = min(r["_valid"] for r in rh3)
        at0 = {r["level"]: r["_point"] for r in rh3 if r["_valid"] == t0}
        rep.p(f"RELHUM: {len(rh3)} messaggi letti, livelli visti {levels[:3]}..{levels[-3:]} "
              f"({len(levels)}), typeOfLevel={rh3[0]['typeOfLevel']}, primo istante {t0:%d %H:%M}")
        if at0:
            low = max(at0)
            rep.p(f"Livello piu' basso letto (numero massimo = piu' vicino al suolo in ICON): {low} -> "
                  f"UR {at0[low]:.1f} %. Ultimi 5 livelli:")
            rep.table(["livello", "UR % sul punto"], [[l, fmt(at0[l], 1)] for l in sorted(at0)[-5:]])
            if T and TD and t0 in T and t0 in TD:
                m = magnus_rh(T[t0], TD[t0])
                rep.check("INFO", "RELHUM livello piu' basso vs Magnus 2 m",
                          f"{at0[low]:.1f} % vs {m:.1f} % (diff {at0[low] - m:+.1f})")
        if len(levels) < 60:
            rep.check("INFO", "RELHUM letta solo in parte", f"{len(levels)} livelli: alza --max-3d per vedere tutto il primo istante")

    # --- vento / raffiche
    rep.h("Vento e raffiche", 3)
    U, V, G = S.get("U_10M"), S.get("V_10M"), S.get("VMAX_10M")
    if U and V:
        rows, ratios = [], []
        for t in U:
            if t in V:
                sp = math.hypot(U[t], V[t])
                dr = (180 + math.degrees(math.atan2(U[t], V[t]))) % 360
                gust = G.get(t) if G else None
                if gust is not None and sp > 0.5:
                    ratios.append(gust / sp)
                rows.append([t.strftime("%d %H:%M"), fmt(U[t], 2), fmt(V[t], 2), fmt(sp * 3.6, 1), round(dr),
                             fmt(gust * 3.6, 1) if gust is not None else "-"])
        rep.table(["valido", "u m/s", "v m/s", "vel km/h", "dir deg", "raffica km/h"], rows[:30])
        uv = rec["U_10M"][0]["_grid"].get("uvRelativeToGrid") if rec.get("U_10M") else None
        gt = g0.get("gridType") if g0 else None
        ang = rotation_angle(g0, P.LAT, P.LON) if g0 else None
        rep.check("INFO", "uvRelativeToGrid / griglia", f"{uv} / {gt}; angolo rotazione griglia nel punto ~{fmt(ang, 2)} deg"
                  if ang is not None else f"{uv} / {gt}; nessun parametro di rotazione")
        if gt == "rotated_ll" and uv == 1 and ang is not None and abs(ang) > 1:
            rep.check("WARN", "vento riferito alla griglia ruotata", f"serve correggere la direzione di ~{abs(ang):.1f} deg")
        if ratios:
            rep.check("OK" if min(ratios) >= 0.95 else "WARN", "raffica >= vento medio",
                      f"rapporto min {min(ratios):.2f}, mediana {np.median(ratios):.2f}")
        if G:
            m = max(G.values())
            rep.check("OK" if 0 <= m < 100 else "WARN", "VMAX_10M in m/s plausibile", f"max {m:.1f} m/s = {m * 3.6:.0f} km/h")

    # --- nuvolosita'
    if "CLCT" in S:
        v = list(S["CLCT"].values())
        allmax = max(r["_max"] for r in rec["CLCT"] if r["_max"] is not None)
        rep.check("OK" if allmax > 1.5 else "WARN", "CLCT in percentuale (non frazione)",
                  f"punto {min(v):.1f}..{max(v):.1f}, massimo sul dominio {allmax:.2f}, units={rec['CLCT'][0]['units']}")

    # --- precipitazione
    rep.h("Precipitazione", 3)
    TP = S.get("TOT_PREC")
    if TP:
        ts = list(TP)
        vals = [TP[t] for t in ts]
        rows = []
        for i, t in enumerate(ts):
            d = vals[i] - vals[i - 1] if i else None
            ex = {k: S[k].get(t) for k in ("RAIN_GSP", "RAIN_CON", "SNOW_GSP", "SNOW_CON") if k in S}
            rows.append([t.strftime("%d %H:%M"), fmt(vals[i], 3), fmt(d, 3) if d is not None else "-"] +
                        [fmt(ex.get(k), 3) if ex.get(k) is not None else "-" for k in ("RAIN_GSP", "RAIN_CON", "SNOW_GSP", "SNOW_CON")])
        rep.table(["valido", "TOT_PREC punto", "delta", "RAIN_GSP", "RAIN_CON", "SNOW_GSP", "SNOW_CON"], rows[:40])
        mono = all(b >= a - 1e-6 for a, b in zip(vals, vals[1:]))
        rep.check("OK" if mono else "WARN", "TOT_PREC monotona crescente (cumulata)",
                  "" if mono else "ci sono cali: probabile reset/finestra mobile, va gestito")
        rep.check("OK" if abs(vals[0]) < 1e-6 else "WARN", "TOT_PREC parte da 0", f"primo valore {vals[0]:.4f} ({ts[0]:%d %H:%M})")
        sty = rec["TOT_PREC"][0]
        rep.check("INFO", "TOT_PREC metadati", f"stepType={sty['stepType']} shortName={sty['shortName']} units={sty['units']} "
                  f"stat={sty['typeOfStatisticalProcessing']} lengthOfTimeRange={sty['lengthOfTimeRange']}")
        last = ts[-1]
        comp = {k: S[k].get(last) for k in ("RAIN_GSP", "RAIN_CON", "SNOW_GSP", "SNOW_CON") if k in S and S[k].get(last) is not None}
        if len(comp) == 4:
            tot = sum(comp.values())
            rep.check("INFO", "TOT_PREC vs somma RAIN+SNOW (ultimo istante, punto)", f"{vals[-1]:.3f} vs {tot:.3f} mm "
                      f"(neve {comp['SNOW_GSP'] + comp['SNOW_CON']:.3f} mm)")
        if sty["units"] not in ("kg m**-2", "kg m-2", "mm"):
            rep.check("WARN", "unita' TOT_PREC inattese", str(sty["units"]))


# ----------------------------------------------------------------------------- 4. end-to-end


def validate_json(rep, path):
    doc = json.load(open(path, encoding="utf-8"))
    rows = doc["hourly"]
    rep.p(f"model={doc.get('model')} run={doc.get('model_run')} righe orarie={len(rows)} "
          f"done={doc['source_meta'].get('done')}")
    rep.check("OK" if doc.get("model") == "ICON_2I_RUC" else "FAIL", "campo model", str(doc.get("model")))
    rep.check("OK" if len(rows) >= 6 else "WARN", "numero ore", str(len(rows)))
    ts = [datetime.strptime(r["time_utc"], "%Y-%m-%dT%H:%MZ") for r in rows]
    gaps = sorted({int((b - a).total_seconds() // 60) for a, b in zip(ts, ts[1:])})
    rep.check("OK" if gaps in ([60], []) else "WARN", "passo orario continuo", f"passi (min): {gaps}")
    bounds = {"temperature_2m": (-30, 50), "relative_humidity_2m": (0, 100), "precipitation_probability": (0, 100),
              "rain": (0, 100), "cloud_cover": (0, 100), "wind_speed_10m": (0, 150),
              "wind_direction_10m": (0, 360), "wind_gusts_10m": (0, 200)}
    stat = []
    for f in P.FIELDS:
        vals = [r.get(f) for r in rows]
        nn = [v for v in vals if v is not None]
        lo, hi = bounds[f]
        ok = len(nn) == len(vals) and nn and lo <= min(nn) and max(nn) <= hi
        stat.append([f, len(vals) - len(nn), fmt(min(nn), 2) if nn else "-", fmt(float(np.mean(nn)), 2) if nn else "-",
                     fmt(max(nn), 2) if nn else "-", "ok" if ok else "CONTROLLA"])
        rep.check("OK" if ok else "WARN", f"campo {f}", f"null={len(vals) - len(nn)}")
    rep.table(["campo", "null", "min", "media", "max", "esito"], stat)
    both = [r for r in rows if r.get("wind_gusts_10m") is not None and r.get("wind_speed_10m") is not None]
    if both:
        frac = sum(r["wind_gusts_10m"] >= r["wind_speed_10m"] for r in both) / len(both)
        rep.check("OK" if frac > .95 else "WARN", "raffica >= vento nel JSON", f"{frac * 100:.0f}% delle ore")
    rep.p("Prime 12 righe del JSON:")
    rep.table(["utc"] + P.FIELDS, [[r["time_utc"]] + [r.get(f) for f in P.FIELDS] for r in rows[:12]])
    rep.p("Cella:")
    rep.code(json.dumps(doc["source_meta"].get("cell"), indent=1))


def section_e2e(rep, base):
    rep.h("Test end-to-end di previsioni.py")
    tmp = tempfile.mkdtemp(prefix="icon2i_e2e_")
    cwd = os.getcwd()
    orig = P.find_latest_run
    P.find_latest_run = lambda: base
    os.chdir(tmp)
    try:
        for n, label in ((1, "primo lancio (download completo)"), (2, "secondo lancio (deve fermarsi subito)")):
            buf, t0, status = io.StringIO(), time.time(), "ok"
            try:
                with contextlib.redirect_stdout(buf):
                    P.main()
            except SystemExit as e:
                status = f"SystemExit: {e}"
            except Exception:
                status = "eccezione:\n" + traceback.format_exc()
            dt = time.time() - t0
            rep.h(f"{label} - {dt:.0f} s", 3)
            rep.code(clean_log(buf.getvalue())[-3500:])
            if status != "ok":
                rep.check("FAIL", f"e2e {label}", status.splitlines()[0])
                rep.code(status)
                break
            rep.check("OK", f"e2e {label}", f"{dt:.0f} s")
            if n == 2:
                rep.check("OK" if dt < 20 and "già completo" in buf.getvalue() else "WARN",
                          "secondo lancio salta il download", f"{dt:.0f} s")
        if os.path.exists("latest.json"):
            rep.h("Validazione latest.json", 3)
            validate_json(rep, "latest.json")
            rep.p(f"File generati in: `{tmp}`")
        else:
            rep.check("FAIL", "latest.json non creato")
    finally:
        os.chdir(cwd)
        P.find_latest_run = orig


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--max-3d", type=int, default=150)
    a = ap.parse_args()

    rep = Report()
    t_start = time.time()
    run = None
    rec = {}
    base = None
    steps = [
        ("ambiente", lambda: section_env(rep)),
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception:
            rep.error(name)

    print("1/5 run disponibili ...")
    try:
        run = section_runs(rep, a.run)
    except Exception:
        rep.error("elenco run")
    if run:
        print("2/5 inventario ...")
        try:
            base = section_inventory(rep, run)
        except Exception:
            rep.error("inventario")
            base = f"{P.ROOT}{run}/"

        print("3/5 analisi GRIB (scarica i file, puo' richiedere qualche minuto) ...")
        rep.h("Analisi GRIB per variabile")
        for f in NEEDED + ([] if a.no_extras else EXTRAS):
            print(f"  - {f}")
            try:
                rec[f] = analyze_folder(rep, base, f, max_msgs=a.max_3d if f in THREE_D else None)
            except Exception:
                rep.error(f"analisi {f}")

        print("4/5 controlli ...")
        try:
            section_checks(rep, rec)
        except Exception:
            rep.error("controlli fisici")

        if not a.skip_e2e:
            print("5/5 test end-to-end ...")
            try:
                section_e2e(rep, base)
            except Exception:
                rep.error("end-to-end")

    rep.h("Fine")
    rep.p(f"Durata totale: {time.time() - t_start:.0f} s")
    name = f"report_icon2i_{run or 'norun'}_{datetime.now():%Y%m%d_%H%M}.md"
    with open(name, "w", encoding="utf-8") as fh:
        fh.write(rep.render("Report diagnostico ICON-2I RUC"))
    n_bad = sum(1 for c in rep.checks if c[0] in ("FAIL", "WARN"))
    print(f"\nFatto. {n_bad} FAIL/WARN su {len(rep.checks)} controlli.\nReport: {os.path.abspath(name)}")


if __name__ == "__main__":
    main()
