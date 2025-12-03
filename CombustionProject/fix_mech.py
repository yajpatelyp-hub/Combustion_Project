import os

# The file name of the mechanism you uploaded
mech_file = 'Base_PRF_Aro_MeOH_EtOH_NOx_GRI.chmech'

print(f"Checking {mech_file} for duplicates...")

if not os.path.exists(mech_file):
    print(f"Error: {mech_file} not found. Make sure you are in the correct folder.")
    exit(1)

with open(mech_file, 'r') as f:
    lines = f.readlines()

new_lines = []
patched_count = 0

i = 0
while i < len(lines):
    line = lines[i]

    # 1. CLEANUP: Fix the previous script's error if present
    # If DUPLICATE is appended to the line end, remove it
    if "DUPLICATE" in line and not line.strip() == "DUPLICATE":
        line = line.replace("DUPLICATE", "").rstrip() + "\n"

    # Normalize for checking
    clean = line.split('!')[0].upper().replace(' ', '').replace('\t', '')

    # Check for essential ingredients
    has_hocho = 'HOCHO' in clean
    has_co2 = 'CO2' in clean
    has_h2 = 'H2' in clean
    has_eq = '=' in clean

    is_target = False
    if has_hocho and has_co2 and has_h2 and has_eq:
        # Target 1: HOCHO + M ...
        if '+M' in clean:
            is_target = True
        # Target 2: HOCHO + H ... (Ensure CO2 is product to avoid mixing up with Reaction 482)
        elif ('+H=' in clean or '+H<' in clean) and 'CO2' in clean:
            is_target = True

    new_lines.append(line)

    if is_target:
        # Check if the NEXT line is already "DUPLICATE"
        # If not, insert it on a NEW line
        next_line_is_dup = False
        if i + 1 < len(lines):
            if "DUPLICATE" in lines[i+1].upper():
                next_line_is_dup = True

        if not next_line_is_dup:
            new_lines.append("DUPLICATE\n")
            patched_count += 1
            print(f"Inserted DUPLICATE keyword after: {line.strip()}")

    i += 1

with open(mech_file, 'w') as f:
    f.writelines(new_lines)

print(f"Done. Patched {patched_count} reactions.")