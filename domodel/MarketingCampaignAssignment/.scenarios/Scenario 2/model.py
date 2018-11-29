from docplex.mp.model import *
from docloud.status import JobSolveStatus
from docplex.mp.conflict_refiner import ConflictRefiner, VarUbConstraintWrapper, VarLbConstraintWrapper
import time
import sys

import pandas as pd
import numpy as np


# Return index values of a multi-index from index name
def helper_get_level_values(df, column_name):
    return df.index.get_level_values(df.index.names.index(column_name))

# Label constraint
def helper_add_labeled_cplex_constraint(mdl, expr, label, context=None, columns=None):
    global expr_counter
    if isinstance(expr, bool):
        pass  # Adding a trivial constraint: if infeasible, docplex will raise an exception it is added to the model
    else:
        expr.name = '_L_EXPR_' + str(len(expr_to_info) + 1)
        if columns:
            ctxt = ", ".join(str(getattr(context, col)) for col in columns)
        else:
            if context:
                ctxt = context.Index if isinstance(context.Index, str) is not None else ", ".join(context.Index)
            else:
                ctxt = None
        expr_to_info[expr.name] = (label, ctxt)
    mdl.add(expr)



# Data model definition for each table
# Data collection: list_of_Campaign ['id', 'max_customers']
# Data collection: list_of_Candidates ['Campaign', 'Customer', 'expected_value', 'line']
# Data collection: list_of_Customer ['id']

# Create a pandas Dataframe for each data table
list_of_Campaign = inputs['Campaign']
list_of_Campaign = list_of_Campaign[['id', 'max customers']].copy()
list_of_Campaign.rename(columns={'id': 'id', 'max customers': 'max_customers'}, inplace=True)
list_of_Candidates = inputs['Candidates']
list_of_Candidates = list_of_Candidates[['Campaign', 'Customer', 'expected value']].copy()
list_of_Candidates.rename(columns={'Campaign': 'Campaign', 'Customer': 'Customer', 'expected value': 'expected_value'}, inplace=True)
list_of_Customer = inputs['Customer']
list_of_Customer = list_of_Customer[['id']].copy()
list_of_Customer.rename(columns={'id': 'id'}, inplace=True)

# Set index when a primary key is defined
list_of_Campaign.set_index('id', inplace=True)
list_of_Campaign.sort_index(inplace=True)
list_of_Campaign.index.name = 'id_of_Campaign'
list_of_Candidates.index.name = 'id_of_Candidates'
list_of_Customer.set_index('id', inplace=True)
list_of_Customer.sort_index(inplace=True)
list_of_Customer.index.name = 'id_of_Customer'

# Create data frame as cartesian product of: Customer x Campaign
list_of_ResourceAssignment = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Customer.index, list_of_Campaign.index), names=['id_of_Customer', 'id_of_Campaign']))


def build_model():
    mdl = Model()

    # Definition of model variables
    list_of_ResourceAssignment['resourceAssignmentVar'] = mdl.binary_var_list(len(list_of_ResourceAssignment))


    # Definition of model
    # Objective cMaximizeAssignmentsAutoSelected-
    # Combine weighted criteria: 
    # 	cMaximizeAssignmentsAutoSelected cMaximizeAssignmentsAutoSelected{
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cSingleCriterionGoal.numericExpr = count( cResourceAssignment[Customer, Campaign]),
    # 	cMaximizeAssignments.assignment = cResourceAssignment[Customer, Campaign]} with weight 5.0
    # 	cMaximizeAssignmentValue cMaximizeAssignmentValue{
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cSingleCriterionGoal.numericExpr = total cResourceAssignment[Customer, Campaign] / Customer / inverse(Candidates.Customer) [Candidates / Campaign is cResourceAssignment[Customer, Campaign] / Campaign] / expected value,
    # 	cMaximizeAssignmentValue.assignment = cResourceAssignment[Customer, Campaign],
    # 	cMaximizeAssignmentValue.assignmentValue = Candidates} with weight 5.0
    join_ResourceAssignment_Candidates_SG1 = list_of_ResourceAssignment.reset_index().merge(list_of_Candidates.reset_index(), left_on=['id_of_Customer'], right_on=['Customer']).set_index(list_of_ResourceAssignment.index.names + list(set(list_of_Candidates.index.names) - set(list_of_ResourceAssignment.index.names)))
    filtered_ResourceAssignment_Candidates_SG1 = join_ResourceAssignment_Candidates_SG1.loc[join_ResourceAssignment_Candidates_SG1.Campaign == helper_get_level_values(join_ResourceAssignment_Candidates_SG1, 'id_of_Campaign')].copy()
    filtered_ResourceAssignment_Candidates_SG1['conditioned_expected_value'] = filtered_ResourceAssignment_Candidates_SG1.resourceAssignmentVar * filtered_ResourceAssignment_Candidates_SG1.expected_value
    agg_ResourceAssignment_Candidates_conditioned_expected_value_SG1 = mdl.sum(filtered_ResourceAssignment_Candidates_SG1.conditioned_expected_value)
    
    mdl.add_kpi(1.0 * (agg_ResourceAssignment_Candidates_conditioned_expected_value_SG1) / 1, publish_name='overall quality of Customer to Campaign assignments according to Candidates')
    
    mdl.maximize( 0
        # Sub Goal cMaximizeAssignmentValue_cMaximizeGoal
        # Maximize overall quality of Customer to Campaign assignments according to Candidates
        + 1.0 * (agg_ResourceAssignment_Candidates_conditioned_expected_value_SG1) / 1
    )
    
    # [ST_1] Constraint : cLimitNumberOfResourcesAssignedToEachActivity_cIterativeRelationalConstraint
    # The number of Customer assignments for each Campaign is less than or equal to max customers
    # Label: CT_1_The_number_of_Customer_assignments_for_each_Campaign_is_less_than_or_equal_to_max_customers
    join_Campaign_ResourceAssignment = list_of_Campaign.join(list_of_ResourceAssignment, how='inner')
    groupbyLevels = [join_Campaign_ResourceAssignment.index.names.index(name) for name in list_of_Campaign.index.names]
    groupby_Campaign_ResourceAssignment = join_Campaign_ResourceAssignment.resourceAssignmentVar.groupby(level=groupbyLevels).sum().to_frame()
    join_Campaign_ResourceAssignment_Campaign = groupby_Campaign_ResourceAssignment.join(list_of_Campaign.max_customers, how='inner')
    for row in join_Campaign_ResourceAssignment_Campaign[join_Campaign_ResourceAssignment_Campaign.max_customers.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.resourceAssignmentVar <= row.max_customers, 'The number of Customer assignments for each Campaign is less than or equal to max customers', row)
    
    # [ST_2] Constraint : cBasicLimitNumberOfActivitiesAssignedToEachResource_cIterativeRelationalConstraint
    # The number of Campaign assignments for each Customer is less than or equal to 1
    # Label: CT_2_The_number_of_Campaign_assignments_for_each_Customer_is_less_than_or_equal_to_1
    groupbyLevels = [list_of_ResourceAssignment.index.names.index(name) for name in list_of_Customer.index.names]
    groupby_ResourceAssignment = list_of_ResourceAssignment.resourceAssignmentVar.groupby(level=groupbyLevels).sum().to_frame()
    for row in groupby_ResourceAssignment.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.resourceAssignmentVar <= 1, 'The number of Campaign assignments for each Customer is less than or equal to 1', row)


    return mdl


def solve_model(mdl):
    mdl.parameters.timelimit = 120
    msol = mdl.solve(log_output=True)
    if not msol:
        print("!!! Solve of the model fails")
        if mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_SOLUTION:
            crefiner = ConflictRefiner()
            conflicts = crefiner.refine_conflict(model, log_output=True)
            export_conflicts(conflicts)
    print 'Solve status: ', mdl.get_solve_status()
    mdl.report()
    return msol


expr_to_info = {}


def export_conflicts(conflicts):
    # Display conflicts in console
    print('Conflict set:')
    list_of_conflicts = pd.DataFrame(columns=['constraint', 'context', 'detail'])
    for conflict, index in zip(conflicts, range(len(conflicts))):
        st = conflict.status
        ct = conflict.element
        label, context = expr_to_info.get(conflict.name, ('N/A', conflict.name))
        label_type = type(conflict.element)
        if isinstance(conflict.element, VarLbConstraintWrapper) \
                or isinstance(conflict.element, VarUbConstraintWrapper):
            ct = conflict.element.get_constraint()

        # Print conflict information in console
        print("Conflict involving constraint: %s, \tfor: %s -> %s" % (label, context, ct))
        list_of_conflicts = list_of_conflicts.append(pd.DataFrame({'constraint': label, 'context': str(context), 'detail': ct},
                                                                  index=[index], columns=['constraint', 'context', 'detail']))

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_conflicts'] = list_of_conflicts


def export_solution(msol):
    start_time = time.time()
    list_of_ResourceAssignment_solution = pd.DataFrame(index=list_of_ResourceAssignment.index)
    list_of_ResourceAssignment_solution['resourceAssignmentVar'] = msol.get_values(list_of_ResourceAssignment.resourceAssignmentVar.values)

    # Filter rows for non-selected assignments
    list_of_ResourceAssignment_solution = list_of_ResourceAssignment_solution[list_of_ResourceAssignment_solution.resourceAssignmentVar > 0.5]

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_ResourceAssignment_solution'] = list_of_ResourceAssignment_solution.reset_index()

    elapsed_time = time.time() - start_time
    print('solution export done in ' + str(elapsed_time) + ' secs')
    return


print('* building wado model')
start_time = time.time()
model = build_model()
elapsed_time = time.time() - start_time
print('model building done in ' + str(elapsed_time) + ' secs')

print('* running wado model')
start_time = time.time()
msol = solve_model(model)
elapsed_time = time.time() - start_time
print('model solve done in ' + str(elapsed_time) + ' secs')
if msol:
    export_solution(msol)
