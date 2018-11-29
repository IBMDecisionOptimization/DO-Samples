from docplex.mp.environment import Environment
from docplex.mp.model import Model
import pandas as pd

N_DAYS = 2
N_PERIODS_PER_DAY = 24*4
N_PERIODS = N_DAYS * N_PERIODS_PER_DAY

df_resources = inputs['resources']
df_demands = inputs['demands']

env = Environment()

mdl = Model("planning")

resources = df_resources['id'].values.tolist()

nb_periods = N_PERIODS

# periods range from 0 to nb_periods excluded
periods = range(0, nb_periods)

# days range from 0 to N_DAYS excluded
days = range(0, N_DAYS)

# start[r,t] is number of resource r starting to work at period t
start = mdl.integer_var_matrix(keys1=resources, keys2=periods, name="start")

# work[r,t] is number of resource r working at period t
work = mdl.integer_var_matrix(keys1=resources, keys2=periods, name="work")

# nr[r] is number of resource r working in total
nr = mdl.integer_var_dict(keys=resources, name="nr")

# nr[r,d] is number of resource r working on day d
nrd = mdl.integer_var_matrix(keys1=resources, keys2=days, name="nrd")

# Organize all decision variables in a DataFrame indexed by 'resources' and 'periods'
df_decision_vars = pd.DataFrame({'start': start, 'work': work})

# Set index names
df_decision_vars.index.names=['resources', 'periods']

# Organize resource decision variables in a DataFrame indexed by 'resources'
df_decision_vars_res = pd.DataFrame({'nr': nr})

# Set index names
df_decision_vars_res.index.names=['resources']

# available per day
for r in resources:
    min_avail = int(df_resources[df_resources.id == r].min_avail)
    max_avail = int(df_resources[df_resources.id == r].max_avail)
    for d in range(N_DAYS):
        mdl.add( mdl.sum(start[r,t] for t in range(d*N_PERIODS_PER_DAY, (d+1)*N_PERIODS_PER_DAY)) == nrd[r,d])
        mdl.add( nrd[r,d] <= nr[r] )
    mdl.add( min_avail <= nr[r] )
    mdl.add(        nr[r] <= max_avail )

# working
for r in resources:
    duration = int(df_resources[df_resources.id == r].duration)*4
    for t in periods:
        mdl.add( mdl.sum(start[r,t2] for t2 in range(max(t-duration,0), t)) == work[r,t])

# work vs demand
for t in periods:
    demand = int(df_demands[df_demands.period == t].demand)
    mdl.add( mdl.sum( work[r,t] for r in resources) >= demand)

if 'fixed' in inputs:
    fixed = inputs['fixed']
    for i in range(len(fixed)):
        r = fixed.get_value(index=i, col='resources')
        p = fixed.get_value(index=i, col='periods')
        v = fixed.get_value(index=i, col='value')
        mdl.add( start[r,p] == v )

total_cost = mdl.sum( int(df_resources[df_resources.id == r].cost)*nr[r] for r in resources)
n_fix_used = nr['fix']
n_temp_used = nr['temp']

mdl.add_kpi(total_cost   , "Total Cost")
mdl.add_kpi(n_fix_used   , "Nb Fix Used")
mdl.add_kpi(n_temp_used   , "Nb Temp Used")
#mdl.add_kpi(lambda x,y:1, "Feasibility", 1))

mdl.minimize(total_cost)

mdl.context.solver.log_output = True

if mdl.solve():
    print("  Feasible " + str(mdl.objective_value))

    df_sol_starts = df_decision_vars.start.apply(lambda v: v.solution_value).unstack(level='resources')
    df_sol_starts = df_sol_starts.stack(level='resources').to_frame()
    df_sol_starts['resources'] = df_sol_starts.index.get_level_values('resources')
    df_sol_starts['periods'] = df_sol_starts.index.get_level_values('periods')
    df_sol_starts.columns = ['value', 'resources', 'periods']
    df_sol_starts = df_sol_starts.reset_index(drop=True)

    df_sol_works = df_decision_vars.work.apply(lambda v: v.solution_value).unstack(level='resources')
    df_sol_works = df_sol_works.stack(level='resources').to_frame()
    df_sol_works['resources'] = df_sol_works.index.get_level_values('resources')
    df_sol_works['periods'] = df_sol_works.index.get_level_values('periods')
    df_sol_works.columns = ['value', 'resources', 'periods']
    df_sol_works = df_sol_works.reset_index(drop=True)

    df_sol_nr = df_decision_vars_res.nr.apply(lambda v: v.solution_value).to_frame()
    df_sol_nr['resources'] = df_sol_nr.index
    df_sol_nr = df_sol_nr.reset_index(drop=True)

    outputs = {}
    outputs['starts'] = df_sol_starts
    outputs['works'] = df_sol_works
    outputs['nr'] = df_sol_nr
else:
    print("  Infeasible")
    outputs = {}
