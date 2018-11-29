from docplex.mp.utils import *
from docplex.cp.model import *
from docplex.cp.expression import _FLOATING_POINT_PRECISION
import time
import operator

import pandas as pd
import numpy as np


pandasDayOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_of_week_calendar_intervals_by_keys = dict()


def helper_get_or_create_day_of_week_calendar(day_of_week_from_as_str, day_of_week_to_as_str=None, time_of_day_from=None, time_of_day_to=None):
    key = 'dayOfWeek_calendar_' + (day_of_week_from_as_str if day_of_week_to_as_str is None else day_of_week_from_as_str + '_' + day_of_week_to_as_str)
    if time_of_day_from is not None and time_of_day_to is not None:
        key += '_' + str(time_of_day_from) + '_' + str(time_of_day_to)
    day_of_week_calendar = day_of_week_calendar_intervals_by_keys.get(key, None)
    if day_of_week_calendar is not None:
        return key, day_of_week_calendar
    try:
        day_of_week_from_int = pandasDayOfWeek.index(day_of_week_from_as_str)
        day_of_week_to_int = day_of_week_from_int if day_of_week_to_as_str is None else pandasDayOfWeek.index(day_of_week_to_as_str)
        cross_horizon_start = all_days_time_series[all_days_time_series.dayofweek == day_of_week_from_int][0] > all_days_time_series[all_days_time_series.dayofweek == day_of_week_to_int][0]
        deltaDay = day_of_week_to_int - day_of_week_from_int
        deltaDay = deltaDay + 7 if deltaDay < 0 else deltaDay
        filtered_all_days_from_ts = pd.Series(all_days_time_series[all_days_time_series.dayofweek == day_of_week_from_int])
        if time_of_day_from is None or time_of_day_to is None:
            day_break_intervals = pd.DataFrame({'start': helper_convert_date_series_to_int(filtered_all_days_from_ts),
                'end': helper_convert_date_series_to_int(filtered_all_days_from_ts + pd.Timedelta(days=1 + deltaDay))},
                columns=['start', 'end'])
            if cross_horizon_start:
                day_break_intervals = pd.DataFrame([helper_convert_date_series_to_int(pd.Series([horizon_start_date, all_days_time_series[all_days_time_series.dayofweek == day_of_week_to_int][0] + pd.Timedelta(days=1)]))], columns=['start', 'end']).append(day_break_intervals)
        else:
            day_break_intervals = pd.DataFrame({'start': helper_convert_date_series_to_int(filtered_all_days_from_ts + time_of_day_from),
                'end': helper_convert_date_series_to_int(filtered_all_days_from_ts + pd.Timedelta(days=deltaDay) + time_of_day_to)},
                columns=['start', 'end'])
            if cross_horizon_start:
                day_break_intervals = pd.DataFrame([helper_convert_date_series_to_int(pd.Series([horizon_start_date, all_days_time_series[all_days_time_series.dayofweek == day_of_week_to_int][0] + time_of_day_to]))], columns=['start', 'end']).append(day_break_intervals)
        day_break_intervals = day_break_intervals.astype(np.int64)
        day_of_week_calendar_intervals_by_keys[key] = day_break_intervals
        return key, day_break_intervals
    except ValueError:
        return 'default', all_calendar_intervals_by_keys['default']  # return default calendar

# Calendars handling
all_calendar_intervals_by_keys = dict()
all_calendar_intervals_by_keys['default'] = pd.DataFrame(columns=['start', 'end'])

internal_calendar_col_id = 'internal_calendar'
default_calendar_id = 'default'


def helper_create_internal_calendar_column(target_df):
    if not internal_calendar_col_id in target_df.columns:
        target_df[internal_calendar_col_id] = default_calendar_id

# Convert type to 'int64'
def helper_int64_convert(arg):
    if pd.__version__ < '0.20.0':
        return arg.astype('int64', raise_on_error=False)
    else:
        return arg.astype('int64', errors='ignore')

# Parse and convert a date Series to an integer Series
# Integer value represents the number of schedule units (time granularity for engine) since horizon start
def helper_convert_date_series_to_int(date_series):
    result = (pd.to_numeric((date_series - horizon_start_date).values) / nanosecs_per_sec * duration_units_per_day * schedUnitPerDurationUnit / secs_per_day)
    result[date_series.isnull().values] = np.nan
    return result

# Parse and convert an integer Series to a date Series
# Integer value represents the number of schedule units (time granularity for engine) since horizon start
def helper_convert_int_series_to_date(sched_int_series):
    return pd.to_datetime(sched_int_series * secs_per_day / duration_units_per_day / schedUnitPerDurationUnit * nanosecs_per_sec + horizon_start_date.value, errors='coerce')

def helper_update_interval_calendars(main_target_df, filtered_target_df, id_col, new_calendar_id, new_calendar_intervals_df):
    flat_target_df = main_target_df.reset_index()
    update_key_ids = len(main_target_df[main_target_df.internal_calendar.isin(filtered_target_df.internal_calendar.unique())]) != len(filtered_target_df)
    grpby = filtered_target_df.reset_index()[[id_col, internal_calendar_col_id]].groupby(internal_calendar_col_id)
    for k, v in grpby:
        current_calendar_intervals = all_calendar_intervals_by_keys[k]
        new_key = k + '+' + new_calendar_id if update_key_ids or k == default_calendar_id else k
        new_calendar_intervals = current_calendar_intervals.append(new_calendar_intervals_df, ignore_index=True)
        all_calendar_intervals_by_keys[new_key] = new_calendar_intervals
        if update_key_ids or k == default_calendar_id:
            flat_target_df.loc[flat_target_df[id_col].isin(v[id_col]), internal_calendar_col_id] = new_key
    main_target_df[internal_calendar_col_id] = flat_target_df.set_index(main_target_df.index.names)[internal_calendar_col_id]

# Convert a duration Series to a Series representing the number of scheduling units
def helper_convert_duration_series_to_scheduling_unit(duration_series, nb_input_data_units_per_day):
    return helper_int64_convert(duration_series * duration_units_per_day * schedUnitPerDurationUnit / nb_input_data_units_per_day)

# Parse and convert a date to an integer
# Integer value represents the number of schedule units (time granularity for engine) since horizon start
def helper_convert_date_to_int(date):
    return int((date - horizon_start_date).value / nanosecs_per_sec * duration_units_per_day * schedUnitPerDurationUnit / secs_per_day)


def helper_parse_and_convert_date_to_int(date_as_str):
    return helper_convert_date_to_int(pd.to_datetime(date_as_str))

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


# Create default calendar
def helper_get_default_calendar():
    calendar = CpoStepFunction()
    calendar.set_value(-INTERVAL_MAX, INTERVAL_MAX, 100)
    return calendar


# Create all calendars (step functions) and assign them to their respective tasks
def helper_build_all_break_calendars():
    all_break_calendars_by_keys = dict()
    for k in all_calendar_intervals_by_keys.keys():
        # Create calendar will 100% availability over planning horizon, then add break intervals
        calendar = helper_get_default_calendar()
        for interval in all_calendar_intervals_by_keys[k].itertuples(index=False):
            calendar.set_value(int(interval.start), int(interval.end), 0)
        all_break_calendars_by_keys[k] = calendar
    return all_break_calendars_by_keys

helper_concept_id_to_index_names_map = {
    'cTask': ['id_of_Activity'],
    'activity': ['id_of_Activity'],
    'cDiscreteResource': ['id_of_Equipment'],
    'equipment': ['id_of_Equipment']}


# Data model definition for each table
# Data collection: list_of_Activity ['Days', 'Id', 'Required_equipment']
# Data collection: list_of_activity_Preceding_activities ['activity', 'Preceding_activities']
# Data collection: list_of_Equipment ['Capacity', 'Id']

# Create a pandas Dataframe for each data table
list_of_Activity = inputs['activity']
list_of_Activity = list_of_Activity[['Days', 'Id', 'Required equipment']].copy()
list_of_Activity.rename(columns={'Days': 'Days', 'Id': 'Id', 'Required equipment': 'Required_equipment'}, inplace=True)
# --- Handling implicit table for multi-valued property
list_of_activity_Preceding_activities = inputs['activity'][['Id', 'Preceding activities']].copy()
list_of_activity_Preceding_activities.rename(columns={'Id': 'activity', 'Preceding activities': 'Preceding_activities'}, inplace=True)
list_of_activity_Preceding_activities.set_index('activity', inplace=True)
list_of_activity_Preceding_activities_split = list_of_activity_Preceding_activities['Preceding_activities'].str.split(',').apply(pd.Series).stack().str.strip()
list_of_activity_Preceding_activities_split.index = list_of_activity_Preceding_activities_split.index.droplevel(-1)
list_of_activity_Preceding_activities = pd.DataFrame(list_of_activity_Preceding_activities_split, columns=['Preceding_activities']).reset_index()
list_of_Equipment = inputs['equipment']
list_of_Equipment = list_of_Equipment[['Capacity', 'Id']].copy()
list_of_Equipment.rename(columns={'Capacity': 'Capacity', 'Id': 'Id'}, inplace=True)

# Set index when a primary key is defined
list_of_Activity.set_index('Id', inplace=True)
list_of_Activity.sort_index(inplace=True)
list_of_Activity.index.name = 'id_of_Activity'
list_of_activity_Preceding_activities.set_index('activity', inplace=True)
list_of_activity_Preceding_activities.sort_index(inplace=True)
list_of_activity_Preceding_activities.index.name = 'id_of_Activity'
list_of_Equipment.set_index('Id', inplace=True)
list_of_Equipment.sort_index(inplace=True)
list_of_Equipment.index.name = 'id_of_Equipment'
# Define time granularity for scheduling
schedUnitPerDurationUnit = 1440  # DurationUnit is days
duration_units_per_day = 1.0


# Define global constants for date to integer conversions
horizon_start_date = pd.to_datetime('Thu Jan 04 00:00:00 UTC 2018')
horizon_end_date = horizon_start_date + pd.Timedelta(days=3650)
nanosecs_per_sec = 1000.0 * 1000 * 1000
secs_per_day = 3600.0 * 24

# Convert all input durations to internal time unit
list_of_Activity['Days'] = helper_convert_duration_series_to_scheduling_unit(list_of_Activity.Days, 1.0)



all_days_time_series = pd.date_range(start=horizon_start_date.date(), end=horizon_end_date.date() + pd.Timedelta(days=1), freq='D')




def build_model():
    mdl = CpoModel()

    # Definition of model variables
    list_of_Activity['interval'] = interval_var_list(len(list_of_Activity), end=(INTERVAL_MIN, INTERVAL_MAX / 4), optional=True)
    list_of_Activity['taskStartVar'] = list_of_Activity.interval.apply(mdl.start_of)
    list_of_Activity['taskEndVar'] = list_of_Activity.interval.apply(mdl.end_of)
    list_of_Activity['taskDurationVar'] = list_of_Activity.interval.apply(mdl.size_of)
    list_of_Activity['taskAbsenceVar'] = 1 - list_of_Activity.interval.apply(mdl.presence_of)
    list_of_Activity['taskPresenceVar'] = list_of_Activity.interval.apply(mdl.presence_of)
    
    # Each activity requires 1 of Required equipment
    list_of_Activity['usage'] = [pulse(row.interval, 1) for row in list_of_Activity.itertuples(index=False)]
    list_of_Equipment_usage = list_of_Activity[['Required_equipment', 'usage']].groupby('Required_equipment').sum()
    list_of_Activity.drop('usage', axis=1, inplace=True)   # Restore input DataFrame


    # Definition of model
    # Objective cMinimizeMakespan-
    # Combine weighted criteria: 
    # 	cMinimizeMakespan cMinimizeMakespan{
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cMinimizeMakespan.taskEnd = cTaskEnd[activity],
    # 	cSingleCriterionGoal.numericExpr = max of count( cTaskEnd[activity]) over cTaskEnd[activity],
    # 	cMinimizeMakespan.task = activity,
    # 	cScaledGoal.scaleFactorExpr = 1} with weight 5.0
    agg_Activity_taskEndVar_SG1 = mdl.max(list_of_Activity.taskEndVar)
    
    kpi_1 = integer_var(name='kpi_1')
    mdl.add(kpi_1 >= 1.0 * (agg_Activity_taskEndVar_SG1 / schedUnitPerDurationUnit) * 1 - 1 + _FLOATING_POINT_PRECISION)
    mdl.add(kpi_1 <= 1.0 * (agg_Activity_taskEndVar_SG1 / schedUnitPerDurationUnit) * 1)
    mdl.add_kpi(kpi_1, name='time to complete all activities')
    
    mdl.add(minimize( 0
        # Sub Goal cMinimizeMakespan_cMinimizeGoal
        # Minimize time to complete all activities
        + 1.0 * (agg_Activity_taskEndVar_SG1 / schedUnitPerDurationUnit) * 1
    ))
    
    # [ST_1] Constraint : cSetFixedDurationSpezProp_cIterativeRelationalConstraint
    # The schedule must respect the duration specified for each activity
    # Label: CT_1_The_schedule_must_respect_the_duration_specified_for_each_activity
    for row in list_of_Activity[list_of_Activity.Days.notnull()].itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, size_of(row.interval, int(row.Days)) == int(row.Days), 'The schedule must respect the duration specified for each activity', row)
    
    # [ST_2] Constraint : cForceTaskPresence_cIterativeRelationalConstraint
    # All activities are present
    # Label: CT_2_All_activities_are_present
    for row in list_of_Activity.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskAbsenceVar != 1, 'All activities are present', row)
    
    # [ST_3] Constraint : cDiscreteResourceCapacityLimitAlwaysConstraintSpez_cDiscreteResourceCapacityLimitConstraint
    # The schedule must comply with the capacity limit defined for each equipment
    # Label: CT_3_The_schedule_must_comply_with_the_capacity_limit_defined_for_each_equipment
    join_list_of_Equipment_usage = list_of_Equipment_usage.join(list_of_Equipment, how='inner')
    for row in join_list_of_Equipment_usage.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.usage <= int(row.Capacity), 'The schedule must comply with the capacity limit defined for each equipment', row)
    
    # [ST_4] Constraint : cTaskPredecessorsNoDelayConstraintDirect_cTaskPredecessorsConstraint
    # Each activity starts after the end of Preceding activities
    # Label: CT_4_Each_activity_starts_after_the_end_of_Preceding_activities
    join_Activity = list_of_Activity.join(list_of_activity_Preceding_activities.Preceding_activities, how='inner')
    join_Activity_2 = join_Activity.reset_index().join(list_of_Activity.interval, on=['Preceding_activities'], rsuffix='_right', how='inner').set_index(['id_of_Activity'])
    for row in join_Activity_2.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, end_before_start(row.interval_right, row.interval), 'Each activity starts after the end of Preceding activities', row)
    
    # [ST_5] Constraint : cTaskPeriodicDayBreakConstraint_cTaskPeriodicDayBreakConstraintAbstract
    # For each activity, add a non-working day every Sunday
    # Label: CT_5_For_each_activity__add_a_non_working_day_every_Sunday
    helper_create_internal_calendar_column(list_of_Activity)
    calendar_id, day_break_intervals = helper_get_or_create_day_of_week_calendar('Sunday')
    helper_update_interval_calendars(list_of_Activity, list_of_Activity, 'id_of_Activity', calendar_id, day_break_intervals)
    
    # Configure all tasks with their respective calendar
    all_break_calendars_by_keys = helper_build_all_break_calendars()
    for row in list_of_Activity[list_of_Activity.internal_calendar != default_calendar_id].itertuples(index=False):
        row.interval.set_intensity(all_break_calendars_by_keys[row.internal_calendar])
        mdl.add(forbid_start(row.interval, all_break_calendars_by_keys[row.internal_calendar]))


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
    list_of_Activity_solution = pd.DataFrame(index=list_of_Activity.index)
    list_of_Activity_solution = list_of_Activity_solution.join(pd.DataFrame([msol.solution[interval] if msol.solution[interval] else (None, None, None) for interval in list_of_Activity.interval], index=list_of_Activity.index, columns=['taskStartVar', 'taskEndVar', 'taskDurationVar']))
    list_of_Activity_solution['taskStartVarDate'] = helper_convert_int_series_to_date(list_of_Activity_solution.taskStartVar)
    list_of_Activity_solution['taskEndVarDate'] = helper_convert_int_series_to_date(list_of_Activity_solution.taskEndVar)
    list_of_Activity_solution.taskStartVar /= schedUnitPerDurationUnit
    list_of_Activity_solution.taskEndVar /= schedUnitPerDurationUnit
    list_of_Activity_solution.taskDurationVar /= schedUnitPerDurationUnit
    list_of_Activity_solution['taskAbsenceVar'] = list_of_Activity.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_absent() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Activity_solution['taskPresenceVar'] = list_of_Activity.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_present() else 0) if msol.solution.get_var_solution(r) else np.NaN)

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_Activity_solution'] = list_of_Activity_solution.reset_index()
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
