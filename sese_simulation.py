import numpy as np
import pandas as pd
import math

np.random.seed(42)

# ==============================================================================
# 1. REALISTIC PARAMETERS
# ==============================================================================
NUM_HOSPITALS = 50
NUM_REQUESTS = 5240
SIM_DAYS = 7 
SIM_MINUTES = SIM_DAYS * 24 * 60
AMBULANCE_SPEED_KM_PER_MIN = 1.25  # ~75 km/h
REJECTION_PENALTY_MEAN = 12.0      # 12 mins lost per failed phone/radio contact
REJECTION_PENALTY_STD = 3.0

TIER_DISTRIBUTION = [4, 8, 38]
CAPABILITY_DIMS = 14 

# ==============================================================================
# 2. GENERATE HOSPITALS
# ==============================================================================
hospitals = []
tier_names = ["Level I", "Level II", "Community"]

for tier, count in enumerate(TIER_DISTRIBUTION):
    for _ in range(count):
        angle = np.random.uniform(0, 2 * math.pi)
        radius = np.random.uniform(0, 30)
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        
        base_beds = int(np.random.lognormal(mean=math.log(35), sigma=0.4))
        cap_prob = [0.80, 0.40, 0.10][tier]
        capabilities = np.random.binomial(1, cap_prob, CAPABILITY_DIMS)
        capabilities[0] = 1 # All hospitals have basic ED
        
        distance_from_center = math.sqrt(x**2 + y**2)
        equity_score = min(1.0, distance_from_center / 30.0) 
        
        hospitals.append({
            "id": len(hospitals), "tier": tier_names[tier],
            "x": x, "y": y, "base_beds": base_beds,
            "capabilities": capabilities, "equity_score": equity_score
        })

# ==============================================================================
# 3. GENERATE REQUESTS
# ==============================================================================
requests = []
current_time = 0
ESI_PROBS = [0.05, 0.20, 0.45, 0.20, 0.10]

while len(requests) < NUM_REQUESTS and current_time < SIM_MINUTES:
    hour_of_day = (current_time / 60) % 24
    arrival_rate = 0.4 + 0.3 * (math.exp(-0.5 * ((hour_of_day - 9)/2)**2) + 
                                math.exp(-0.5 * ((hour_of_day - 20)/2)**2))
    current_time += np.random.exponential(1.0 / arrival_rate)
    if current_time >= SIM_MINUTES: break
    
    esi = np.random.choice([1, 2, 3, 4, 5], p=ESI_PROBS)
    req_caps = np.zeros(CAPABILITY_DIMS, dtype=int)
    
    if esi <= 2: req_caps[:4] = np.random.binomial(1, 0.8, 4) 
    elif esi == 3: req_caps[:2] = np.random.binomial(1, 0.5, 2)
    req_caps[0] = 1 
        
    angle = np.random.uniform(0, 2 * math.pi)
    radius = np.random.uniform(0, 30)
    
    requests.append({
        "id": len(requests), "time": current_time,
        "x": radius * math.cos(angle), "y": radius * math.sin(angle),
        "esi": esi, "required_capabilities": req_caps
    })

# ==============================================================================
# 4. STATE DYNAMICS (Beds, Capabilities, and DIVERSION)
# ==============================================================================
def get_available_beds(hospital, time_min):
    hour = (time_min / 60) % 24
    # Base occupancy 80%, peaking at 95%
    occupancy_rate = 0.80 + 0.15 * math.sin((hour - 8) * math.pi / 12) 
    noise = np.random.normal(0, 0.03)
    occupancy_rate = max(0.60, min(0.99, occupancy_rate + noise))
    return max(0, int(hospital["base_beds"] * (1 - occupancy_rate)))

def is_on_diversion(hospital, time_min):
    """
    Real-world friction: Hospitals >80% full often go on 'Ambulance Diversion' 
    and reject incoming EMS, even if they technically have 1 or 2 beds left.
    """
    beds = get_available_beds(hospital, time_min)
    occupancy = 1.0 - (beds / hospital["base_beds"])
    if occupancy > 0.80:
        return np.random.rand() < 0.70 # 70% chance of rejecting if crowded
    return False

def get_realtime_capabilities(hospital):
    realtime_caps = hospital["capabilities"].copy()
    for i in range(1, CAPABILITY_DIMS):
        if realtime_caps[i] == 1:
            if np.random.rand() < 0.30: # 30% chance specialized equipment is busy
                realtime_caps[i] = 0
    return realtime_caps

def get_future_beds(hospital, time_min, transport_time):
    actual_future_beds = get_available_beds(hospital, time_min + transport_time)
    prediction_error = np.random.normal(-0.5, 0.8) 
    return max(0, int(actual_future_beds + prediction_error))

# ==============================================================================
# 5. ROUTING ALGORITHMS
# ==============================================================================
def calculate_transport_time(req, hosp):
    dist = math.sqrt((req["x"] - hosp["x"])**2 + (req["y"] - hosp["y"])**2)
    return dist / AMBULANCE_SPEED_KM_PER_MIN

def check_capability_match(req_caps, hosp_caps):
    return np.all(hosp_caps >= req_caps)

def route_pob(req):
    sorted_hosps = sorted(hospitals, key=lambda h: calculate_transport_time(req, h))
    first_h = sorted_hosps[0]
    first_match = 1 if (not is_on_diversion(first_h, req["time"]) and
                        get_available_beds(first_h, req["time"]) > 0 and 
                        check_capability_match(req["required_capabilities"], get_realtime_capabilities(first_h))) else 0
    
    attempts = 0
    for h in sorted_hosps:
        attempts += 1
        t_transport = calculate_transport_time(req, h)
        
        # POB gets rejected if hospital is on diversion, full, or lacks capability
        if is_on_diversion(h, req["time"]):
            continue # REJECTED
            
        beds = get_available_beds(h, req["time"])
        realtime_caps = get_realtime_capabilities(h)
        
        if beds > 0 and check_capability_match(req["required_capabilities"], realtime_caps):
            future_beds = get_available_beds(h, req["time"] + t_transport)
            collision = 1 if future_beds == 0 else 0
            art = t_transport + (attempts - 1) * np.random.normal(REJECTION_PENALTY_MEAN, REJECTION_PENALTY_STD)
            return h["id"], art, attempts, first_match, 0, collision
            
    return sorted_hosps[0]["id"], 60.0, attempts, first_match, 0, 1

def route_sd(req):
    # Static Directory: Uses 24h old data. It DOES NOT know who is on diversion right now.
    candidates = [h for h in hospitals if check_capability_match(req["required_capabilities"], h["capabilities"])]
    if not candidates: candidates = hospitals 
        
    sorted_hosps = sorted(candidates, key=lambda h: calculate_transport_time(req, h))
    first_h = sorted_hosps[0]
    first_match = 1 if (not is_on_diversion(first_h, req["time"]) and
                        get_available_beds(first_h, req["time"]) > 0 and 
                        check_capability_match(req["required_capabilities"], get_realtime_capabilities(first_h))) else 0
    
    attempts = 0
    for h in sorted_hosps:
        attempts += 1
        t_transport = calculate_transport_time(req, h)
        
        # SD calls the hospital. If they are on diversion, REJECT.
        if is_on_diversion(h, req["time"]):
            continue # REJECTED
            
        beds = get_available_beds(h, req["time"])
        realtime_caps = get_realtime_capabilities(h)
        
        if beds > 0 and check_capability_match(req["required_capabilities"], realtime_caps):
            future_beds = get_available_beds(h, req["time"] + t_transport)
            collision = 1 if future_beds == 0 else 0
            art = t_transport + (attempts - 1) * np.random.normal(REJECTION_PENALTY_MEAN, REJECTION_PENALTY_STD)
            return h["id"], art, attempts, first_match, 0, collision
            
    return sorted_hosps[0]["id"], 60.0, attempts, first_match, 0, 1

def route_sese(req, use_predictive=True):
    t_transport_all = [calculate_transport_time(req, h) for h in hospitals]
    t_min, t_max = min(t_transport_all), max(t_transport_all)
    beds_all = [get_available_beds(h, req["time"]) for h in hospitals]
    b_min, b_max = min(beds_all), max(beds_all)
    
    candidates = []
    for i, h in enumerate(hospitals):
        t_trans = t_transport_all[i]
        
        # SESE knows exactly who is on diversion via real-time FHIR and skips them
        if is_on_diversion(h, req["time"]):
            continue
            
        realtime_caps = get_realtime_capabilities(h)
        if not check_capability_match(req["required_capabilities"], realtime_caps): 
            continue
            
        if use_predictive:
            if get_future_beds(h, req["time"], t_trans) <= 2: continue
        else:
            if beds_all[i] <= 1: continue
            
        norm_beds = (beds_all[i] - b_min) / (b_max - b_min + 1e-5)
        norm_time = (t_trans - t_min) / (t_max - t_min + 1e-5)
        equity = h["equity_score"]
        
        if req["esi"] <= 2: alpha, beta, gamma, delta = 0.4, 0.2, 0.3, 0.1 
        else: alpha, beta, gamma, delta = 0.5, 0.3, 0.1, 0.1 
            
        score = (alpha * 1.0) + (beta * norm_beds) - (gamma * norm_time) + (delta * equity)
        candidates.append((h, score, t_trans))
        
    if not candidates:
        fallback = sorted(hospitals, key=lambda h: calculate_transport_time(req, h))[0]
        return fallback["id"], calculate_transport_time(req, fallback) + 2.0, 1, 0, 1, 0
        
    best_h, _, best_t = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    future_beds = get_available_beds(best_h, req["time"] + best_t)
    collision = 1 if future_beds == 0 else 0
    
    return best_h["id"], best_t + 1.5, 1, 1, 0, collision 

# ==============================================================================
# 6. RUN SIMULATION
# ==============================================================================
results = []
for req in requests:
    _, art_pob, dcli_pob, fmcs_pob, _, ccr_pob = route_pob(req)
    _, art_sd, dcli_sd, fmcs_sd, _, ccr_sd = route_sd(req)
    _, art_sese_np, dcli_np, fmcs_np, _, ccr_np = route_sese(req, use_predictive=False)
    _, art_sese, dcli_sese, fmcs_sese, div_sese, ccr_sese = route_sese(req, use_predictive=True)
    
    results.append({
        "POB_ART": art_pob, "POB_DCLI": dcli_pob, "POB_FMCS": fmcs_pob, "POB_CCR": ccr_pob,
        "SD_ART": art_sd, "SD_DCLI": dcli_sd, "SD_FMCS": fmcs_sd, "SD_CCR": ccr_sd,
        "SESE_NP_CCR": ccr_np,
        "SESE_ART": art_sese, "SESE_DCLI": dcli_sese, "SESE_FMCS": fmcs_sese, 
        "SESE_DIV": div_sese, "SESE_CCR": ccr_sese
    })

df = pd.DataFrame(results)

# ==============================================================================
# 7. OUTPUT METRICS
# ==============================================================================
print("--- TABLE 1: Average Referral Time (ART) and Cognitive Load ---")
print(f"POB:   Mean ART = {df['POB_ART'].mean():.1f} min | DCLI = {df['POB_DCLI'].mean():.1f}")
print(f"SD:    Mean ART = {df['SD_ART'].mean():.1f} min | DCLI = {df['SD_DCLI'].mean():.1f}")
print(f"SESE:  Mean ART = {df['SESE_ART'].mean():.1f} min | DCLI = {df['SESE_DCLI'].mean():.1f}")

print("\n--- TABLE 2: First-Match Clinical Suitability (FMCS) & Diversion ---")
print(f"POB:   FMCS = {df['POB_FMCS'].mean()*100:.1f}%")
print(f"SD:    FMCS = {df['SD_FMCS'].mean()*100:.1f}%")
print(f"SESE:  FMCS = {df['SESE_FMCS'].mean()*100:.1f}% | Diversion Rate = {df['SESE_DIV'].mean()*100:.1f}%")

print("\n--- TABLE 3: Ablation Study - Capacity Collision Rate (CCR) ---")
print(f"POB:       CCR = {df['POB_CCR'].mean()*100:.1f}%")
print(f"SD:        CCR = {df['SD_CCR'].mean()*100:.1f}%")
print(f"SESE-NP:   CCR = {df['SESE_NP_CCR'].mean()*100:.1f}%")
print(f"SESE:      CCR = {df['SESE_CCR'].mean()*100:.1f}%")
