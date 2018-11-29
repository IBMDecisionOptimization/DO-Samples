from docplex.mp.utils import *
from docplex.cp.model import *
from docplex.cp.expression import _FLOATING_POINT_PRECISION
import time
import operator

import pandas as pd
import numpy as np


# Convert type to 'int64'
def helper_int64_convert(arg):
    if pd.__version__ < '0.20.0':
        return arg.astype('int64', raise_on_error=False)
    else:
        return arg.astype('int64', errors='ignore')

# Parse and convert an integer Series to a date Series
# Integer value represents the number of schedule units (time granularity for engine) since horizon start
def helper_convert_int_series_to_date(sched_int_series):
    return pd.to_datetime(sched_int_series * secs_per_day / duration_units_per_day / schedUnitPerDurationUnit * nanosecs_per_sec + horizon_start_date.value, errors='coerce')

# Return index values of a multi-index from index name
def helper_get_level_values(df, column_name):
    return df.index.get_level_values(df.index.names.index(column_name))

# Convert a duration Series to a Series representing the number of scheduling units
def helper_convert_duration_series_to_scheduling_unit(duration_series, nb_input_data_units_per_day):
    return helper_int64_convert(duration_series * duration_units_per_day * schedUnitPerDurationUnit / nb_input_data_units_per_day)

# Label constraint
expr_counter = 1
def helper_add_labeled_cpo_constraint(mdl, expr, label, context=None, columns=None):
    global expr_counter
    if isinstance(expr, bool):
        pass  # Adding a trivial constraint: if infeasible, docplex will raise an exception it is added to the model
    else:
        expr.name = '_L_EXPR_' + str(expr_counter)
        expr_counter += 1
        if columns:
            ctxt = ", ".join(str(getattr(context, col)) for col in columns)
        else:
            if context:
                ctxt = context.Index if isinstance(context.Index, str) is not None else ", ".join(context.Index)
            else:
                ctxt = None
        expr_to_info[expr.name] = (label, ctxt)
    mdl.add(expr)

def helper_get_index_names_for_type(dataframe, type):
    if not is_pandas_dataframe(dataframe):
        return None
    return [name for name in dataframe.index.names if name in helper_concept_id_to_index_names_map.get(type, [])]


helper_concept_id_to_index_names_map = {
    'cTask': ['id_of_Activity'],
    'Subcontractor': ['id_of_Subcontractor'],
    'Activity': ['id_of_Activity'],
    'cUnaryResource': ['id_of_Subcontractor']}


# Data model definition for each table
# Data collection: list_of_Activity ['Duration_in_days', 'Activity']
# Data collection: list_of_Activity_Possible_Subcontractors ['Activity', 'Possible_Subcontractors']
# Data collection: list_of_Activity_Preceding_activities ['Activity', 'Preceding_activities']
# Data collection: list_of_Subcontractor ['Name']

# Create a pandas Dataframe for each data table
list_of_Activity = inputs['Activity']
list_of_Activity = list_of_Activity[['Duration in days', 'Activity']].copy()
list_of_Activity.rename(columns={'Duration in days': 'Duration_in_days', 'Activity': 'Activity'}, inplace=True)
# --- Handling implicit table for multi-valued property
list_of_Activity_Possible_Subcontractors = inputs['Activity'][['Activity', 'Possible Subcontractors']].copy()
list_of_Activity_Possible_Subcontractors.rename(columns={'Activity': 'Activity', 'Possible Subcontractors': 'Possible_Subcontractors'}, inplace=True)
list_of_Activity_Possible_Subcontractors.set_index('Activity', inplace=True)
list_of_Activity_Possible_Subcontractors_split = list_of_Activity_Possible_Subcontractors['Possible_Subcontractors'].str.split(',').apply(pd.Series).stack().str.strip()
list_of_Activity_Possible_Subcontractors_split.index = list_of_Activity_Possible_Subcontractors_split.index.droplevel(-1)
list_of_Activity_Possible_Subcontractors = pd.DataFrame(list_of_Activity_Possible_Subcontractors_split, columns=['Possible_Subcontractors']).reset_index()
# --- Handling implicit table for multi-valued property
list_of_Activity_Preceding_activities = inputs['Activity'][['Activity', 'Preceding activities']].copy()
list_of_Activity_Preceding_activities.rename(columns={'Activity': 'Activity', 'Preceding activities': 'Preceding_activities'}, inplace=True)
list_of_Activity_Preceding_activities.set_index('Activity', inplace=True)
list_of_Activity_Preceding_activities_split = list_of_Activity_Preceding_activities['Preceding_activities'].str.split(',').apply(pd.Series).stack().str.strip()
list_of_Activity_Preceding_activities_split.index = list_of_Activity_Preceding_activities_split.index.droplevel(-1)
list_of_Activity_Preceding_activities = pd.DataFrame(list_of_Activity_Preceding_activities_split, columns=['Preceding_activities']).reset_index()
list_of_Subcontractor = inputs['Subcontractor']
list_of_Subcontractor = list_of_Subcontractor[['Name']].copy()
list_of_Subcontractor.rename(columns={'Name': 'Name'}, inplace=True)

# Set index when a primary key is defined
list_of_Activity.set_index('Activity', inplace=True)
list_of_Activity.sort_index(inplace=True)
list_of_Activity.index.name = 'id_of_Activity'
list_of_Activity_Possible_Subcontractors.set_index('Activity', inplace=True)
list_of_Activity_Possible_Subcontractors.sort_index(inplace=True)
list_of_Activity_Possible_Subcontractors.index.name = 'id_of_Activity'
list_of_Activity_Preceding_activities.set_index('Activity', inplace=True)
list_of_Activity_Preceding_activities.sort_index(inplace=True)
list_of_Activity_Preceding_activities.index.name = 'id_of_Activity'
list_of_Subcontractor.set_index('Name', inplace=True)
list_of_Subcontractor.sort_index(inplace=True)
list_of_Subcontractor.index.name = 'id_of_Subcontractor'
# Define time granularity for scheduling
schedUnitPerDurationUnit = 1440  # DurationUnit is days
duration_units_per_day = 1.0


# Define global constants for date to integer conversions
horizon_start_date = pd.to_datetime('Wed Jan 03 00:00:00 UTC 2018')
horizon_end_date = horizon_start_date + pd.Timedelta(days=3650)
nanosecs_per_sec = 1000.0 * 1000 * 1000
secs_per_day = 3600.0 * 24

# Convert all input durations to internal time unit
list_of_Activity['Duration_in_days'] = helper_convert_duration_series_to_scheduling_unit(list_of_Activity.Duration_in_days, 1.0)


# Create data frame as cartesian product of: Activity x Subcontractor
list_of_SchedulingAssignment = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Activity.index, list_of_Subcontractor.index), names=['id_of_Activity', 'id_of_Subcontractor']))




def build_model():
    mdl = CpoModel()

    # Definition of model variables
    list_of_SchedulingAssignment['interval'] = interval_var_list(len(list_of_SchedulingAssignment), end=(INTERVAL_MIN, INTERVAL_MAX / 4), optional=True)
    list_of_SchedulingAssignment['schedulingAssignmentVar'] = list_of_SchedulingAssignment.interval.apply(mdl.presence_of)
    list_of_Activity['interval'] = interval_var_list(len(list_of_Activity), end=(INTERVAL_MIN, INTERVAL_MAX / 4), optional=True)
    list_of_Activity['taskStartVar'] = list_of_Activity.interval.apply(mdl.start_of)
    list_of_Activity['taskEndVar'] = list_of_Activity.interval.apply(mdl.end_of)
    list_of_Activity['taskDurationVar'] = list_of_Activity.interval.apply(mdl.size_of)
    list_of_Activity['taskAbsenceVar'] = 1 - list_of_Activity.interval.apply(mdl.presence_of)
    list_of_Activity['taskPresenceVar'] = list_of_Activity.interval.apply(mdl.presence_of)


    # Definition of model
    # Objective cMinimizeMakespan-
    # Combine weighted criteria: 
    # 	cMinimizeMakespan cMinimizeMakespan{
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cMinimizeMakespan.taskEnd = cTaskEnd[Activity],
    # 	cSingleCriterionGoal.numericExpr = max of count( cTaskEnd[Activity]) over cTaskEnd[Activity],
    # 	cMinimizeMakespan.task = Activity,
    # 	cScaledGoal.scaleFactorExpr = 1} with weight 5.0
    agg_Activity_taskEndVar_SG1 = mdl.max(list_of_Activity.taskEndVar)
    
    kpi_1 = integer_var(name='kpi_1')
    mdl.add(kpi_1 >= 1.0 * (agg_Activity_taskEndVar_SG1 / schedUnitPerDurationUnit) * 1 - 1 + _FLOATING_POINT_PRECISION)
    mdl.add(kpi_1 <= 1.0 * (agg_Activity_taskEndVar_SG1 / schedUnitPerDurationUnit) * 1)
    mdl.add_kpi(kpi_1, name='time to complete all Activities')
    
    mdl.add(minimize( 0
        # Sub Goal cMinimizeMakespan_cMinimizeGoal
        # Minimize time to complete all Activities
        + 1.0 * (agg_Activity_taskEndVar_SG1 / schedUnitPerDurationUnit) * 1
    ))
    
    # [ST_1] Constraint : cLimitNumberOfResourcesAssignedToEachActivitySched_cIterativeRelationalConstraint
    # The number of Subcontractor assignments for each Activity is equal to 1
    # Label: CT_1_The_number_of_Subcontractor_assignments_for_each_Activity_is_equal_to_1
    join_Activity = list_of_Activity.join(list_of_SchedulingAssignment, rsuffix='_right', how='inner')
    groupbyLevels = [join_Activity.index.names.index(name) for name in list_of_Activity.index.names]
    groupby_Activity = join_Activity.schedulingAssignmentVar.groupby(level=groupbyLevels).sum().to_frame()
    for row in groupby_Activity.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.schedulingAssignmentVar == 1, 'The number of Subcontractor assignments for each Activity is equal to 1', row)
    
    # [ST_2] Constraint : cSetFixedDurationSpezProp_cIterativeRelationalConstraint
    # The schedule must respect the duration specified for each Activity
    # Label: CT_2_The_schedule_must_respect_the_duration_specified_for_each_Activity
    for row in list_of_Activity[list_of_Activity.Duration_in_days.notnull()].itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, size_of(row.interval, int(row.Duration_in_days)) == int(row.Duration_in_days), 'The schedule must respect the duration specified for each Activity', row)
    
    # [ST_3] Constraint : cForceTaskPresence_cIterativeRelationalConstraint
    # All Activities are present
    # Label: CT_3_All_Activities_are_present
    for row in list_of_Activity.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskAbsenceVar != 1, 'All Activities are present', row)
    
    # [ST_4] Constraint : cTaskPredecessorsNoDelayConstraintDirect_cTaskPredecessorsConstraint
    # Each Activity starts after the end of Preceding activities
    # Label: CT_4_Each_Activity_starts_after_the_end_of_Preceding_activities
    join_Activity = list_of_Activity.join(list_of_Activity_Preceding_activities.Preceding_activities, how='inner')
    join_Activity_2 = join_Activity.reset_index().join(list_of_Activity.interval, on=['Preceding_activities'], rsuffix='_right', how='inner').set_index(['id_of_Activity'])
    for row in join_Activity_2.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, end_before_start(row.interval_right, row.interval), 'Each Activity starts after the end of Preceding activities', row)
    
    # [ST_5] Constraint : cDefineCompatibleResources_cCategoryCompatibilityConstraintOnPair
    # For each Subcontractor to Activity assignment, assigned Subcontractors must be in Possible Subcontractors for assigned Activities
    # Label: CT_5_For_each_Subcontractor_to_Activity_assignment__assigned_Subcontractors_must_be_in_Possible_Subcontractors_for_assigned_Activities
    join_Activity = list_of_Activity.join(list_of_Activity_Possible_Subcontractors.Possible_Subcontractors, how='inner')
    join_SchedulingAssignment = list_of_SchedulingAssignment.reset_index().set_index(['id_of_Activity']).join(join_Activity.Possible_Subcontractors, how='inner').reset_index().set_index(['id_of_Activity', 'id_of_Subcontractor'])
    filtered_SchedulingAssignment = join_SchedulingAssignment.loc[join_SchedulingAssignment.Possible_Subcontractors == helper_get_level_values(join_SchedulingAssignment, 'id_of_Subcontractor')].copy()
    drop_list_of_SchedulingAssignment = list_of_SchedulingAssignment.drop(labels=[-1], level='id_of_Subcontractor')
    helper_add_labeled_cpo_constraint(mdl, mdl.sum(drop_list_of_SchedulingAssignment.schedulingAssignmentVar[~drop_list_of_SchedulingAssignment.index.isin(filtered_SchedulingAssignment.index.values)]) == 0, 'For each Subcontractor to Activity assignment, assigned Subcontractors must be in Possible Subcontractors for assigned Activities')
    
    # Scheduling internal structure
    groupbyLevels = [list_of_SchedulingAssignment.index.names.index(name) for name in list_of_Activity.index.names]
    groupby_SchedulingAssignment = list_of_SchedulingAssignment.interval.groupby(level=groupbyLevels).apply(list).to_frame(name='interval')
    join_SchedulingAssignment = groupby_SchedulingAssignment.join(list_of_Activity.interval, rsuffix='_right', how='inner')
    for row in join_SchedulingAssignment.itertuples(index=False):
        mdl.add(synchronize(row.interval_right, row.interval))
    
    # link presence if not alternative
    groupbyLevels = [list_of_SchedulingAssignment.index.names.index(name) for name in list_of_Activity.index.names]
    groupby_SchedulingAssignment = list_of_SchedulingAssignment.schedulingAssignmentVar.groupby(level=groupbyLevels).agg(lambda l: mdl.max(l.tolist())).to_frame()
    join_SchedulingAssignment = groupby_SchedulingAssignment.join(list_of_Activity.taskPresenceVar, how='inner')
    for row in join_SchedulingAssignment.itertuples(index=False):
        mdl.add(row.schedulingAssignmentVar <= row.taskPresenceVar)
    
    # no overlap
    groupbyLevels = [list_of_SchedulingAssignment.index.names.index(name) for name in list_of_Subcontractor.index.names]
    groupby_SchedulingAssignment = list_of_SchedulingAssignment.interval.groupby(level=groupbyLevels).apply(list).to_frame(name='interval')
    for row in groupby_SchedulingAssignment.reset_index().itertuples(index=False):
        mdl.add(no_overlap(row.interval))


    return mdl


def solve_model(mdl):
    params = CpoParameters()
    params.TimeLimit = 120
    # Call to custom code to update parameters value
    custom_code.update_solver_params(params)
    # Update parameters value based on environment variables definition
    cpo_param_env_prefix = 'ma.cpo.'
    cpo_params = [name[4:] for name in dir(CpoParameters) if name.startswith('set_')]
    for param in cpo_params:
        env_param = cpo_param_env_prefix + param
        param_value = get_environment().get_parameter(env_param)
        if param_value:
            # Updating parameter value
            print("Updated value for parameter %s = %s" % (param, param_value))
            params[param] = param_value

    solver = CpoSolver(mdl, params=params, trace_log=True)
    try:
        for i, msol in enumerate(solver):
            ovals = msol.get_objective_values()
            print("Objective values: {}".format(ovals))
            for k, v in msol.get_kpis().iteritems():
                print k, '-->', v
            export_solution(msol)
            if ovals is None:
                break  # No objective: stop after first solution
        # If model is infeasible, invoke conflict refiner to return
        if solver.get_last_solution().get_solve_status() == SOLVE_STATUS_INFEASIBLE:
            conflicts = solver.refine_conflict()
            export_conflicts(conflicts)
    except CpoException as e:
        # Solve has been aborted from an external action
        print('An exception has been raised: %s' % str(e))
        raise e


expr_to_info = {}


def export_conflicts(conflicts):
    # Display conflicts in console
    print conflicts
    list_of_conflicts = pd.DataFrame(columns=['constraint', 'context', 'detail'])
    for item, index in zip(conflicts.member_constraints, range(len(conflicts.member_constraints))):
        label, context = expr_to_info.get(item.name, ('N/A', item.name))
        constraint_detail = expression._to_string(item)
        # Print conflict information in console
        print("Conflict involving constraint: %s, \tfor: %s -> %s" % (label, context, constraint_detail))
        list_of_conflicts = list_of_conflicts.append(pd.DataFrame({'constraint': label, 'context': str(context), 'detail': constraint_detail},
                                                                  index=[index], columns=['constraint', 'context', 'detail']))

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_conflicts'] = list_of_conflicts


def export_solution(msol):
    start_time = time.time()
    list_of_SchedulingAssignment_solution = pd.DataFrame(index=list_of_SchedulingAssignment.index)
    list_of_SchedulingAssignment_solution['schedulingAssignmentVar'] = list_of_SchedulingAssignment.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_present() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Activity_solution = pd.DataFrame(index=list_of_Activity.index)
    list_of_Activity_solution = list_of_Activity_solution.join(pd.DataFrame([msol.solution[interval] if msol.solution[interval] else (None, None, None) for interval in list_of_Activity.interval], index=list_of_Activity.index, columns=['taskStartVar', 'taskEndVar', 'taskDurationVar']))
    list_of_Activity_solution['taskStartVarDate'] = helper_convert_int_series_to_date(list_of_Activity_solution.taskStartVar)
    list_of_Activity_solution['taskEndVarDate'] = helper_convert_int_series_to_date(list_of_Activity_solution.taskEndVar)
    list_of_Activity_solution.taskStartVar /= schedUnitPerDurationUnit
    list_of_Activity_solution.taskEndVar /= schedUnitPerDurationUnit
    list_of_Activity_solution.taskDurationVar /= schedUnitPerDurationUnit
    list_of_Activity_solution['taskAbsenceVar'] = list_of_Activity.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_absent() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Activity_solution['taskPresenceVar'] = list_of_Activity.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_present() else 0) if msol.solution.get_var_solution(r) else np.NaN)

    # Filter rows for non-selected assignments
    list_of_SchedulingAssignment_solution = list_of_SchedulingAssignment_solution[list_of_SchedulingAssignment_solution.schedulingAssignmentVar > 0.5]

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_Activity_solution'] = list_of_Activity_solution.reset_index()
        outputs['list_of_SchedulingAssignment_solution'] = list_of_SchedulingAssignment_solution.reset_index()
        custom_code.post_process_solution(msol, outputs)

    elapsed_time = time.time() - start_time
    print('solution export done in ' + str(elapsed_time) + ' secs')
    return


# Import custom code definition if module exists
try:
    from custom_code import CustomCode
    custom_code = CustomCode(globals())
except ImportError:
    # Create a dummy anonymous object for custom_code
    custom_code = type('', (object,), {'preprocess': (lambda *args: None),
                                       'update_model': (lambda *args: None),
                                       'update_solver_params': (lambda *args: None),
                                       'post_process_solution': (lambda *args: None)})()

# Custom pre-process
custom_code.preprocess()

print('* building wado model')
start_time = time.time()
model = build_model()

# Model customization
custom_code.update_model(model)

elapsed_time = time.time() - start_time
print('model building done in ' + str(elapsed_time) + ' secs')

print('* running wado model')
start_time = time.time()
solve_model(model)
elapsed_time = time.time() - start_time
print('solve + export of all intermediate solutions done in ' + str(elapsed_time) + ' secs')
