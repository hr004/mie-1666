
model.addConstr(B_r[w] * data['assembly_time'] == R[w], name=f"regular_baskets_{w+1}")
model.addConstr(B_o[w] * data['assembly_time'] == O[w], name=f"overtime_baskets_{w+1}")
