"""
test_full_flow.py -- Step-through test: nearest-ball selection.
Press ENTER to advance. Run: python test/computer/test_full_flow.py
If stale-pyc errors, delete controller/__pycache__ and retry.

Scenario: Robot(30,60)cm 45deg | W1(80,60) W2(150,15) W3(40,105) OB(130,90) Cross(100,35)
"""
import sys, math, time
from unittest.mock import MagicMock
from dataclasses import dataclass
sys.path.insert(0, ".")

# --- Fix stale pyc for calibration_tracker ---
import controller.calibration_tracker as _ct
if not hasattr(_ct, 'calibration_angle_left'):
    _ct.calibration_angle_left = _ct.CalibrationTracker(_ct.DEGREES_PER_ROTATION_LEFT)
    _ct.calibration_angle_right = _ct.CalibrationTracker(_ct.DEGREES_PER_ROTATION_RIGHT)

# --- Constants ---
FW, FH = 900, 600
FW_CM, FH_CM = 180.0, 120.0
SCX, SCY = FW/FW_CM, FH/FH_CM  # 5.0
PX_ROT, DEG_ROT = 47.0, 25.0

# --- Simulated robot ---
class SimRobot:
    def __init__(s, x, y, a):
        s.x, s.y, s.angle, s.collected = x, y, a, 0
    @property
    def px(s): return (s.x*SCX, s.y*SCY)
    @property
    def cm(s): return (s.x, s.y)
    def drive(s, rot):
        d = rot*PX_ROT; r = math.radians(s.angle)
        s.x = max(0,min(s.x + d/SCX*math.cos(r), FW_CM))
        s.y = max(0,min(s.y + d/SCY*math.sin(r), FH_CM))
    def turn(s, rot, d):
        deg = rot*DEG_ROT
        s.angle += deg if d=="RIGHT" else -deg
        s.angle = (s.angle+180)%360-180
    def reverse(s, rot):
        d = rot*PX_ROT; r = math.radians(s.angle)
        s.x = max(0,min(s.x - d/SCX*math.cos(r), FW_CM))
        s.y = max(0,min(s.y - d/SCY*math.sin(r), FH_CM))

@dataclass
class Ball:
    label: str; cm: tuple; color: str; collected: bool = False
    @property
    def px(s): return (s.cm[0]*SCX, s.cm[1]*SCY)

# --- Scenario ---
sim = SimRobot(30.0, 60.0, 45.0)
balls = [
    Ball("W1",(80,60),"white"), Ball("W2",(150,15),"white"),
    Ball("W3",(40,105),"white"), Ball("OB",(130,90),"orange"),
]
CROSS_CM = (100.0, 35.0)
CROSS_PX = (CROSS_CM[0]*SCX, CROSS_CM[1]*SCY)
step_count = 0

# --- Mock EV3 ---
def _mt(rot, d):
    sim.turn(rot, d)
    print(f"    [MOCK] turn {d} {rot:.2f}rot ({rot*DEG_ROT:.1f}deg) -> heading={sim.angle:.1f}deg")
def _md(rot):
    o=sim.cm; sim.drive(rot)
    print(f"    [MOCK] drive {rot:.2f}rot ({rot*PX_ROT:.0f}px) -> ({o[0]:.1f},{o[1]:.1f})->({sim.x:.1f},{sim.y:.1f})cm")
def _mr(rot):
    o=sim.cm; sim.reverse(rot)
    print(f"    [MOCK] reverse {rot:.2f}rot -> ({o[0]:.1f},{o[1]:.1f})->({sim.x:.1f},{sim.y:.1f})cm")
def _mc():
    uc = [b for b in balls if not b.collected]
    if uc:
        c = min(uc, key=lambda b: math.dist(sim.cm,b.cm)); c.collected=True; sim.collected+=1
        print(f"    [MOCK] collect {c.label} at ({c.cm[0]:.0f},{c.cm[1]:.0f})cm  total={sim.collected}")
def _mgo(): print("    [MOCK] gate OPEN")
def _mgc(): print("    [MOCK] gate CLOSE")

mk = MagicMock()
mk.turn.side_effect=_mt; mk.drive.side_effect=_md; mk.reverse.side_effect=_mr
mk.collect.side_effect=_mc; mk.gate_open.side_effect=_mgo; mk.gate_close.side_effect=_mgc
mk.stop.side_effect=lambda: None
sys.modules["controller.ev3_controller"] = mk

# --- Force-load state_machine from source (handles stale pyc + truncated mount) ---
import types as _t
with open("controller/state_machine.py","rb") as _f:
    _src = _f.read().replace(b"\x00",b"")
# Append helpers that may be truncated off the end of the file
_src += b"""
import math as _math
from controller.calibration_tracker import calibration_pixels as _cp
from controller.calibration_tracker import calibration_angle_left as _cal
from controller.calibration_tracker import calibration_angle_right as _car
if '_distance_px' not in dir():
    def _distance_px(a, b):
        if a is None or b is None: return 0.0
        return _math.hypot(a[0]-b[0], a[1]-b[1])
if '_px_to_rotations' not in dir():
    def _px_to_rotations(px): return px / _cp.ratio
if '_angle_to_rotations' not in dir():
    def _angle_to_rotations(err):
        t = _car if err > 0 else _cal
        return abs(err) / t.ratio * 0.6
"""
_c = compile(_src, "controller/state_machine.py", "exec")
_m = _t.ModuleType("controller.state_machine"); _m.__file__="controller/state_machine.py"; _m.__package__="controller"
sys.modules["controller.state_machine"] = _m; exec(_c, _m.__dict__)

from controller.state_machine import GolfBotController, State
from controller.commands import Command
from controller.route_manager import RouteManager, RouteTarget
from controller.navigation import angle_to_target, angle_error

# --- Patch missing class methods (truncated source) ---
def _pi(cls,n,fn):
    if not hasattr(cls,n): setattr(cls,n,fn)
def _tr(s, ns): print(f"[FSM] {s.state.name} -> {ns.name}"); s.state = ns
def _ed(s, p, r):
    s._cal.record_drive(p.px, r)
    import controller.ev3_controller as _r; _r.drive(r); s._pose.invalidate()
def _et(s, p, r, d):
    s._cal.record_turn(p.angle, r, d.name)
    import controller.ev3_controller as _r; _r.turn(r, d.name); s._pose.invalidate()
def _he(s, p, tc):
    return angle_error(p.angle, angle_to_target(p.pos, tc))
def _hp(s, p, tp):
    return angle_error(p.angle, angle_to_target(p.px, tp))
def _rb(s):
    import controller.ev3_controller as _r
    _r.gate_open(); time.sleep(0.1); _r.gate_close()
    s._transition(State.DONE); return Command.RELEASE
for n,f in [('_transition',_tr),('_execute_drive',_ed),('_execute_turn',_et),
            ('_heading_error_to',_he),('_heading_error_to_px',_hp),('_release_balls',_rb)]:
    setattr(GolfBotController, n, f)

# --- Patch RouteManager: closest ball ---
def _closest(self, rp, wb):
    sb = sorted(wb, key=lambda b: math.dist(rp, b))
    self._route = list(sb); self._white_count = len(wb)
    print(f"    [ROUTE] CLOSEST-BALL over {len(wb)} whites:")
    for i,b in enumerate(sb):
        print(f"            [{i+1}] ({b[0]:.0f},{b[1]:.0f})cm d={math.dist(rp,b):.1f}")
for attr in ('_compute_white_route','_compute'):
    if hasattr(RouteManager, attr): setattr(RouteManager, attr, _closest); break

# --- World builder ---
def build_world():
    wc = [b.cm for b in balls if not b.collected and b.color=="white"]
    wp = [b.px for b in balls if not b.collected and b.color=="white"]
    ob = next((b for b in balls if not b.collected and b.color=="orange"), None)
    return {"robot":sim.cm,"robot_px":sim.px,"robot_angle":sim.angle,
            "white_balls":wc,"white_balls_px":wp,
            "ob":ob.cm if ob else None,"ob_px":ob.px if ob else None,
            "cross":CROSS_CM,"cross_px":CROSS_PX}

# --- Display ---
def field_map():
    W,H = 60,20
    g = [['.' for _ in range(W)] for _ in range(H)]
    def p(cm,ch):
        x=int(cm[0]/FW_CM*(W-1)); y=int(cm[1]/FH_CM*(H-1))
        g[max(0,min(y,H-1))][max(0,min(x,W-1))] = ch
    p(CROSS_CM,'X'); p((FW_CM,FH_CM/2),'G')
    for b in balls:
        if not b.collected: p(b.cm, 'O' if b.color=="orange" else 'W')
    p(sim.cm,'R')
    print(f"\n    +{'-'*W}+")
    for r in g: print(f"    |{''.join(r)}|")
    print(f"    +{'-'*W}+")
    print(f"    R=Robot W=White O=Orange X=Cross G=Goal")

def status(ctrl):
    print(f"\n{'-'*70}")
    print(f"  ROBOT: ({sim.x:.1f},{sim.y:.1f})cm px=({sim.px[0]:.0f},{sim.px[1]:.0f}) heading={sim.angle:.1f}deg")
    print(f"  STATE: {ctrl.state.name}")
    rem = [b for b in balls if not b.collected]
    nw = sum(1 for b in rem if b.color=="white")
    no = sum(1 for b in rem if b.color=="orange")
    print(f"  BALLS: {nw}w {no}o remaining ({sim.collected} collected)")
    for b in rem:
        d=math.dist(sim.cm,b.cm)
        br=math.degrees(math.atan2(b.cm[1]-sim.y, b.cm[0]-sim.x))
        e=(br-sim.angle+180)%360-180
        print(f"    {b.label}: ({b.cm[0]:.0f},{b.cm[1]:.0f})cm d={d:.1f} bear={br:.1f} err={e:.1f}deg")
    if ctrl._locked_target:
        t=ctrl._locked_target; print(f"  LOCK: ({t.cm[0]:.0f},{t.cm[1]:.0f})cm px=({t.px[0]:.0f},{t.px[1]:.0f})")
    else: print(f"  LOCK: none")
    print(f"  CROSS: ({CROSS_CM[0]:.0f},{CROSS_CM[1]:.0f})cm d={math.dist(sim.cm,CROSS_CM):.1f}")

def wait():
    global step_count; step_count += 1
    try: input(f"\n  >> Step {step_count} -- press ENTER...")
    except (KeyboardInterrupt, EOFError): print("\nAborted."); sys.exit(0)

# --- Main ---
def run():
    print(f"\n{'='*70}")
    print("  GOLFBOT FULL FLOW -- Closest Ball | ENTER to step | Ctrl+C quit")
    print(f"{'='*70}")
    print(f"\n  Robot:({sim.x:.0f},{sim.y:.0f})cm {sim.angle:.0f}deg")
    for b in balls: print(f"  {b.label}: ({b.cm[0]:.0f},{b.cm[1]:.0f})cm [{b.color}]")
    print(f"  Cross:({CROSS_CM[0]:.0f},{CROSS_CM[1]:.0f})cm  Goal:(180,60)cm")
    field_map()
    ctrl = GolfBotController()
    prev = ctrl.state
    for _ in range(200):
        # Auto-skip settle window (pose cache blackout after motor commands)
        if time.time() < ctrl._pose._valid_after:
            r = ctrl._pose._valid_after - time.time()
            if r > 0: time.sleep(r + 0.01)
        w = build_world()
        status(ctrl); field_map(); wait()
        print(f"\n  -- {ctrl.state.name} --")
        cmd = ctrl.update(w)
        print(f"\n  => cmd={cmd.name} state={ctrl.state.name}")
        if ctrl.state != prev: print(f"  ** {prev.name} -> {ctrl.state.name}")
        prev = ctrl.state
        if ctrl.state == State.DONE:
            print(f"\n{'='*70}\n  DONE! collected={sim.collected} pos=({sim.x:.1f},{sim.y:.1f})cm steps={step_count}\n{'='*70}")
            break
    else: print(f"\n  [!] Max steps reached.")

if __name__ == "__main__":
    run()
