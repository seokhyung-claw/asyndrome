from asyndrome import Schedule, CSSCode, Brisbane

code = CSSCode.from_file("qecc/self-dual-bbcode.json")
schedule = Schedule.from_file("qecc/self-dual-bbcode/alpha-bp_lsd.json")
print(schedule.max_tick)
xrate, zrate = schedule.evaluate(code, "bp_lsd", Brisbane(), 100000)
print(1 - (1 - xrate) * (1 - zrate))