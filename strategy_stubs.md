useful for all to consider:
max_search_radius (simulation.toml config option, default 0.07 rad)

dual spiral
- both satellites spiral from center to max_search_radius, then backwards from max to center
- speeds related by irrational ratio (eg sqrt2)

raster
- A starts at top of circle formed by max_search_radius
- it scans horizontally briefly across then moves down diagonally
- it scans the now larger row, etc
- meanwhile B has the same pattern but scans vertical columns of the search circle
- speeds may need to be irrationally related

rosettes
- each satellite follows the motion:
- x=Acos(w_1 * t), y=Acos(w_2 * t) where w_1/w_2 irrational


random walk
- pick dir, move one beam width, hold
- repeat

random curve
- every t step, diverge randomly up to a small max_angle (plus or minus)
- for both random ones make sure to keep within the search range

nested spiral
- one moves in a spiral pattern but holds every beam width
- during the hold, the other spirals up to max_radius (which may be smaller than max_search_radius for efficiency)
- first one advances, second repeats looking for the beam

golden angle spiral
- r_n = c*sqrt(n)
- theta_n = n * 137.508 deg