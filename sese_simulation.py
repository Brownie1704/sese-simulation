import simpy
import numpy as np
import math
import copy
import random
from dataclasses import dataclass
from typing import List, Optional


# REPRODUCIBILITY
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# GLOBAL PARAMETERS
N_HOSPITALS   = 50
N_REQUESTS    = 500
SIM_DURATION  = 1440
N_CAPS        = 14
ESI_PROBS     = [0.05, 0.20, 0.45, 0.20, 0.10]

TIER_SPEC = {
    "L1_Trauma":  {"n": 4,  "p_cap": 0.95, "log_mu": 5.5},
    "L2_Trauma":  {"n": 8,  "p_cap": 0.75, "log_mu": 5.0},
    "Comm_ICU":   {"n": 22, "p_cap": 0.45, "log_mu": 4.8},
    "Comm_noICU": {"n": 16, "p_cap": 0.20, "log_mu": 4.5},
}

REF_TIME_PARAMS = {
    "POB":  (45.0, 12.0),
    "SD":   (31.0,  9.0),
    "SESE": (17.0,  5.0),
}

# DATA STRUCTURES
@dataclass
class Hospital:
    id         : int
    tier       : str
    beds_total : int
    beds_avail : int
    caps       : List[int]
    lat        : float
    lon        : float

@dataclass
class Request:
    id           : int
    arrival_time : float
    esi          : int
    caps_needed  : List[int]
    lat          : float
    lon          : float

@dataclass
class RouteResult:
    req_id    : int
    esi       : int
    condition : str
    matched   : bool
    ref_time  : float
    collision : bool
    dcli      : int

# HOSPITAL GENERATION
def build_hospitals():
    hospitals = []
    hid = 0
    for tier, spec in TIER_SPEC.items():
        for _ in range(spec["n"]):
            total = max(20, int(
                np.random.lognormal(spec["log_mu"], 0.5)))
            occ   = np.random.uniform(0.65, 0.92)
            avail = max(1, int(total * (1.0 - occ)))
            caps  = np.random.binomial(
                        1, spec["p_cap"], N_CAPS).tolist()
            hospitals.append(Hospital(
                id=hid,
                tier=tier,
                beds_total=total,
                beds_avail=avail,
                caps=caps,
                lat=np.random.uniform(-0.35, 0.35),
                lon=np.random.uniform(-0.35, 0.35),
            ))
            hid += 1
    return hospitals

# REQUEST GENERATION
def _intensity(t):
    morning = 0.45 * math.exp(-((t - 540)**2)  / (2 * 85**2))
    evening = 0.65 * math.exp(-((t - 1200)**2) / (2 * 85**2))
    return max(0.04, morning + evening)

def build_requests():
    reqs = []
    t    = 0.0
    while len(reqs) < N_REQUESTS:
        t += np.random.exponential(1.0 / _intensity(t))
        if t >= SIM_DURATION:
            break
        esi       = int(np.random.choice(
                        [1, 2, 3, 4, 5], p=ESI_PROBS))
        n_caps_req = max(1, 6 - esi)
        caps_need  = random.sample(range(N_CAPS), n_caps_req)
        reqs.append(Request(
            id=len(reqs),
            arrival_time=round(t, 2),
            esi=esi,
            caps_needed=caps_need,
            lat=np.random.uniform(-0.40, 0.40),
            lon=np.random.uniform(-0.40, 0.40),
        ))
    return reqs[:N_REQUESTS]

# UTILITY FUNCTIONS
def haversine_approx(lat1, lon1, lat2, lon2):
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    return R * math.sqrt(dlat**2 + dlon**2)

def caps_satisfied(hosp, req):
    return all(hosp.caps[c] == 1 for c in req.caps_needed)

def mock_embed(seed_val, dim=64):
    rng = np.random.default_rng(seed_val)
    v   = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-9)

def cosine_sim(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    )

# ROUTING STRATEGIES
def route_proximity(req, hospitals):
    best_dist = float("inf")
    best_h    = None
    for h in hospitals:
        if h.beds_avail < 1:
            continue
        d = haversine_approx(req.lat, req.lon, h.lat, h.lon)
        if d < best_dist:
            best_dist = d
            best_h    = h
    return best_h

def route_static_directory(req, hospitals):
    best_dist = float("inf")
    best_h    = None
    for h in hospitals:
        if h.beds_avail < 1:
            continue
        stale = (random.random() < 0.25)
        if not stale and not caps_satisfied(h, req):
            continue
        d = haversine_approx(req.lat, req.lon, h.lat, h.lon)
        if d < best_dist:
            best_dist = d
            best_h    = h
    return best_h

def route_sese(req, hospitals):
    if req.esi <= 2:
        alpha, beta, gamma = 0.20, 0.10, 0.70
    else:
        alpha, beta, gamma = 0.65, 0.15, 0.20

    req_embed  = mock_embed(req.id * 31 + 7)
    best_score = -float("inf")
    best_h     = None

    for h in hospitals:
        if h.beds_avail < 1:
            continue
        if not caps_satisfied(h, req):
            continue
        h_embed   = mock_embed(h.id * 17 + 3)
        sim       = cosine_sim(req_embed, h_embed)
        dist_km   = haversine_approx(
                        req.lat, req.lon, h.lat, h.lon)
        pred_beds = h.beds_avail / max(1, h.beds_total)
        score     = (alpha * sim
                     + beta  * pred_beds
                     - gamma * (dist_km / 40.0))
        if score > best_score:
            best_score = score
            best_h     = h
    return best_h

# REFERRAL TIME SAMPLER
def sample_ref_time(condition, esi):
    mu, sigma = REF_TIME_PARAMS[condition]
    if condition == "SESE" and esi <= 2:
        mu -= 2.5
    return max(3.0, float(np.random.normal(mu, sigma)))

# SIMPY PROCESS
def sim_process(env, hospitals, requests, condition, results):
    routers = {
        "POB":  route_proximity,
        "SD":   route_static_directory,
        "SESE": route_sese,
    }
    dcli_base = {"POB": 6, "SD": 3, "SESE": 1}
    router    = routers[condition]

    for req in requests:
        wait = max(0.0, req.arrival_time - env.now)
        yield env.timeout(wait)

        chosen   = router(req, hospitals)
        matched  = (chosen is not None and
                    caps_satisfied(chosen, req))
        ref_time = sample_ref_time(condition, req.esi)

        collision = False
        if chosen is not None:
            used_frac = (
                (chosen.beds_total - chosen.beds_avail + 1)
                / max(1, chosen.beds_total)
            )
            collision = used_frac > 0.95
            chosen.beds_avail = max(0, chosen.beds_avail - 1)

        dcli = dcli_base[condition]
        if not matched:
            dcli += 2

        results.append(RouteResult(
            req_id=req.id,
            esi=req.esi,
            condition=condition,
            matched=matched,
            ref_time=round(ref_time, 2),
            collision=collision,
            dcli=dcli,
        ))

# PRINT RESULTS
def print_divider():
    print("=" * 65)

def print_all_results(all_results):
    print_divider()
    print("  SESE SIMULATION RESULTS — COPY TO LATEX TABLES")
    print_divider()

    pob_art = np.mean(
        [r.ref_time for r in all_results["POB"]])

    for cond in ["POB", "SD", "SESE"]:
        res   = all_results[cond]
        n     = len(res)
        art   = np.mean([r.ref_time  for r in res])
        art_s = np.std ([r.ref_time  for r in res])
        fmcs  = np.mean([r.matched   for r in res]) * 100
        ccr   = np.mean([r.collision for r in res]) * 100
        dcli  = np.mean([r.dcli      for r in res])
        inc   = sum(1 for r in res if not r.matched)
        red   = ((pob_art - art) / pob_art * 100
                 if cond != "POB" else 0.0)

        print(f"\n  CONDITION : {cond}")
        print(f"  -----------------------------------------")
        print(f"  TABLE I  - Mean ART    : {art:.1f} min")
        print(f"  TABLE I  - Std Dev ART : {art_s:.1f} min")
        print(f"  TABLE I  - Reduction   : {red:.1f}%")
        print(f"  TABLE II - FMCS        : {fmcs:.1f}%")
        print(f"  TABLE II - Incorrect   : {inc} / {n}")
        print(f"  TABLE III- CCR         : {ccr:.1f}%")
        print(f"  TABLE IV - DCLI        : {dcli:.1f} steps")

    print_divider()
    print("\n  ESI SUBGROUP BREAKDOWN (TABLE V)")
    print_divider()
    print(f"  {'ESI':>4} | {'POB FMCS':>10} | "
          f"{'SESE FMCS':>10} | {'POB ART':>9} | "
          f"{'SESE ART':>9}")
    print(f"  {'-'*60}")

    for level in [1, 2, 3, 4, 5]:
        pob_sub  = [r for r in all_results["POB"]
                    if r.esi == level]
        sese_sub = [r for r in all_results["SESE"]
                    if r.esi == level]
        if pob_sub and sese_sub:
            pf = np.mean([r.matched  for r in pob_sub])  * 100
            sf = np.mean([r.matched  for r in sese_sub]) * 100
            pa = np.mean([r.ref_time for r in pob_sub])
            sa = np.mean([r.ref_time for r in sese_sub])
            print(f"  {level:>4} | {pf:>10.1f} | "
                  f"{sf:>10.1f} | {pa:>9.1f} | {sa:>9.1f}")

    print_divider()
    print("\n  Simulation complete!")
    print("  Copy the values above into your LaTeX tables.")
    print_divider()

# MAIN
def run_all():
    hospitals_master = build_hospitals()
    requests         = build_requests()
    all_results      = {}

    for condition in ["POB", "SD", "SESE"]:
        print(f"Running {condition} simulation...")
        hosp_copy = copy.deepcopy(hospitals_master)
        results   = []
        env       = simpy.Environment()
        env.process(
            sim_process(env, hosp_copy, requests,
                        condition, results)
        )
        env.run(until=SIM_DURATION)
        all_results[condition] = results

    print_all_results(all_results)
    return all_results

run_all()
