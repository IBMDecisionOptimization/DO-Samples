from docplex.mp.model import *
from docplex.mp.utils import *
from docloud.status import JobSolveStatus
from docplex.mp.conflict_refiner import ConflictRefiner, VarUbConstraintWrapper, VarLbConstraintWrapper
import time
import sys
import operator

import pandas as pd
import numpy as np


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

def helper_get_index_names_for_type(dataframe, type):
    if not is_pandas_dataframe(dataframe):
        return None
    return [name for name in dataframe.index.names if name in helper_concept_id_to_index_names_map.get(type, [])]


helper_concept_id_to_index_names_map = {
    'cSalesProfit': ['id_of_Marginal_profit'],
    'product': ['id_of_Product'],
    'plant_product_cost': ['id_of_Plant_product_cost'],
    'plant_product_capacity': ['id_of_Plant_product_capacity'],
    'cTimeBucket': ['id_of_Month'],
    'firm_sales': ['id_of_Firm_sales'],
    'demand': ['id_of_Demand'],
    'initial_inventory': ['id_of_Initial_inventory'],
    'marginal_profit': ['id_of_Marginal_profit'],
    'market': ['id_of_Market'],
    'plant_month_capacity': ['id_of_Plant_month_capacity'],
    'month': ['id_of_Month'],
    'cProduct': ['id_of_Product'],
    'plant': ['id_of_Plant'],
    'cLocation': ['id_of_Plant'],
    'cDemand': ['id_of_Demand', 'id_of_Firm_sales'],
    'cInitialCustomerInventory': ['id_of_Initial_inventory'],
    'cPlanningCustomer': ['id_of_Market'],
    'parameters': ['id_of_Parameters'],
    'cProductionCost': ['id_of_Plant_product_cost']}


# Data model definition for each table
# Data collection: list_of_Demand ['Market', 'Month', 'Product', 'Value', 'line']
# Data collection: list_of_Firm_sales ['Market', 'Month', 'Product', 'Value', 'line']
# Data collection: list_of_Initial_inventory ['Market', 'Product', 'Value', 'line']
# Data collection: list_of_Marginal_profit ['Market', 'Month', 'Product', 'Value', 'line']
# Data collection: list_of_Market ['Name']
# Data collection: list_of_Month ['Next', 'Name']
# Data collection: list_of_Parameters ['ID', 'first_month']
# Data collection: list_of_Plant ['Name']
# Data collection: list_of_Plant_month_capacity ['Month', 'Plant', 'Value', 'line']
# Data collection: list_of_Plant_product_capacity ['Plant', 'Product', 'Value']
# Data collection: list_of_Plant_product_cost ['Plant', 'Product', 'Value', 'line']
# Data collection: list_of_Product ['Name']

# Create a pandas Dataframe for each data table
list_of_Demand = inputs['demand']
list_of_Demand = list_of_Demand[['Market', 'Month', 'Product', 'Value']].copy()
list_of_Demand.rename(columns={'Market': 'Market', 'Month': 'Month', 'Product': 'Product', 'Value': 'Value'}, inplace=True)
list_of_Firm_sales = inputs['firm_sales']
list_of_Firm_sales = list_of_Firm_sales[['Market', 'Month', 'Product', 'Value']].copy()
list_of_Firm_sales.rename(columns={'Market': 'Market', 'Month': 'Month', 'Product': 'Product', 'Value': 'Value'}, inplace=True)
list_of_Initial_inventory = inputs['initial_inventory']
list_of_Initial_inventory = list_of_Initial_inventory[['Market', 'Product', 'Value']].copy()
list_of_Initial_inventory.rename(columns={'Market': 'Market', 'Product': 'Product', 'Value': 'Value'}, inplace=True)
list_of_Marginal_profit = inputs['marginal_profit']
list_of_Marginal_profit = list_of_Marginal_profit[['Market', 'Month', 'Product', 'Value']].copy()
list_of_Marginal_profit.rename(columns={'Market': 'Market', 'Month': 'Month', 'Product': 'Product', 'Value': 'Value'}, inplace=True)
list_of_Market = inputs['market']
list_of_Market = list_of_Market[['Name']].copy()
list_of_Market.rename(columns={'Name': 'Name'}, inplace=True)
list_of_Month = inputs['month']
list_of_Month = list_of_Month[['Next', 'Name']].copy()
list_of_Month.rename(columns={'Next': 'Next', 'Name': 'Name'}, inplace=True)
list_of_Parameters = inputs['parameters']
list_of_Parameters = list_of_Parameters[['ID', 'first_month']].copy()
list_of_Parameters.rename(columns={'ID': 'ID', 'first_month': 'first_month'}, inplace=True)
list_of_Plant = inputs['plant']
list_of_Plant = list_of_Plant[['Name']].copy()
list_of_Plant.rename(columns={'Name': 'Name'}, inplace=True)
list_of_Plant_month_capacity = inputs['plant_month_capacity']
list_of_Plant_month_capacity = list_of_Plant_month_capacity[['Month', 'Plant', 'Value']].copy()
list_of_Plant_month_capacity.rename(columns={'Month': 'Month', 'Plant': 'Plant', 'Value': 'Value'}, inplace=True)
list_of_Plant_product_capacity = inputs['plant_product_capacity']
list_of_Plant_product_capacity = list_of_Plant_product_capacity[['Plant', 'Product', 'Value']].copy()
list_of_Plant_product_capacity.rename(columns={'Plant': 'Plant', 'Product': 'Product', 'Value': 'Value'}, inplace=True)
list_of_Plant_product_cost = inputs['plant_product_cost']
list_of_Plant_product_cost = list_of_Plant_product_cost[['Plant', 'Product', 'Value']].copy()
list_of_Plant_product_cost.rename(columns={'Plant': 'Plant', 'Product': 'Product', 'Value': 'Value'}, inplace=True)
list_of_Product = inputs['product']
list_of_Product = list_of_Product[['Name']].copy()
list_of_Product.rename(columns={'Name': 'Name'}, inplace=True)

# Set index when a primary key is defined
list_of_Demand.index.name = 'id_of_Demand'
list_of_Firm_sales.index.name = 'id_of_Firm_sales'
list_of_Initial_inventory.index.name = 'id_of_Initial_inventory'
list_of_Marginal_profit.index.name = 'id_of_Marginal_profit'
list_of_Market.set_index('Name', inplace=True)
list_of_Market.sort_index(inplace=True)
list_of_Market.index.name = 'id_of_Market'
list_of_Month.set_index('Name', inplace=True)
list_of_Month.sort_index(inplace=True)
list_of_Month.index.name = 'id_of_Month'
list_of_Parameters.set_index('ID', inplace=True)
list_of_Parameters.sort_index(inplace=True)
list_of_Parameters.index.name = 'id_of_Parameters'
list_of_Plant.set_index('Name', inplace=True)
list_of_Plant.sort_index(inplace=True)
list_of_Plant.index.name = 'id_of_Plant'
list_of_Plant_month_capacity.index.name = 'id_of_Plant_month_capacity'
list_of_Plant_product_capacity.set_index('Value', inplace=True)
list_of_Plant_product_capacity.sort_index(inplace=True)
list_of_Plant_product_capacity.index.name = 'id_of_Plant_product_capacity'
list_of_Plant_product_cost.index.name = 'id_of_Plant_product_cost'
list_of_Product.set_index('Name', inplace=True)
list_of_Product.sort_index(inplace=True)
list_of_Product.index.name = 'id_of_Product'


# Create data frame as cartesian product of: Plant x Month x Product
list_of_Production = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Plant.index, list_of_Month.index, list_of_Product.index), names=['id_of_Plant', 'id_of_Month', 'id_of_Product']))
# Create data frame as cartesian product of: Market x Month x Product
list_of_ExecutedSales = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Market.index, list_of_Month.index, list_of_Product.index), names=['id_of_Market', 'id_of_Month', 'id_of_Product']))
# Create data frame as cartesian product of: Market x Month x Product
list_of_CustomerInventory = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Market.index, list_of_Month.index, list_of_Product.index), names=['id_of_Market', 'id_of_Month', 'id_of_Product']))
# Create data frame as cartesian product of: Plant x Month x Product
list_of_Inventory = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Plant.index, list_of_Month.index, list_of_Product.index), names=['id_of_Plant', 'id_of_Month', 'id_of_Product']))
# Create data frame as cartesian product of: Plant x Market x Month x Product
list_of_Delivery = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Plant.index, list_of_Market.index, list_of_Month.index, list_of_Product.index), names=['id_of_Plant', 'id_of_Market', 'id_of_Month', 'id_of_Product']))




def build_model():
    mdl = Model()

    # Definition of model variables
    list_of_Production['productionVar'] = mdl.continuous_var_list(len(list_of_Production))
    list_of_ExecutedSales['executedSalesVar'] = mdl.continuous_var_list(len(list_of_ExecutedSales))
    list_of_CustomerInventory['customerInventoryVar'] = mdl.continuous_var_list(len(list_of_CustomerInventory))
    list_of_Inventory['inventoryVar'] = mdl.continuous_var_list(len(list_of_Inventory))
    list_of_Delivery['deliveryVar'] = mdl.continuous_var_list(len(list_of_Delivery))


    # Definition of model
    # Objective (Guided) Maximize MARKET sales-
    # Combine weighted criteria: 
    # 	cMaximizeExecutedSales cMaximizeExecutedSales{
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cMaximizeExecutedSales.customer = market,
    # 	cSingleCriterionGoal.numericExpr = count( cExecutedSales[market, month, product]),
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cMaximizeExecutedSales.executedSales = cExecutedSales[market, month, product]} with weight 5.0
    # 	cMinimizeTotalCustomerInventory cMinimizeTotalCustomerInventory{
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cMinimizeTotalCustomerInventory.customer = market,
    # 	cSingleCriterionGoal.numericExpr = count( cCustomerInventory[market, month, product]),
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cMinimizeTotalCustomerInventory.customerInventory = cCustomerInventory[market, month, product]} with weight 5.0
    # 	cMinimizeProductionCost cMinimizeProductionCost{
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cSingleCriterionGoal.numericExpr = total plant_product_cost [plant_product_cost is joined to cProduction[plant, month, product]] / Value over cProduction[plant, month, product],
    # 	cMinimizeProductionCost.productionCost = plant_product_cost,
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cMinimizeProductionCost.production = cProduction[plant, month, product]} with weight 5.0
    # 	cMaximizeExecutedSalesProfit cMaximizeExecutedSalesProfit{
    # 	cMaximizeExecutedSalesProfit.salesProfit = marginal_profit,
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cMaximizeExecutedSalesProfit.executedSales = cExecutedSales[market, month, product]} with weight 5.0
    agg_ExecutedSales_executedSalesVar_SG1 = mdl.sum(list_of_ExecutedSales.executedSalesVar)
    agg_CustomerInventory_customerInventoryVar_SG2 = mdl.sum(list_of_CustomerInventory.customerInventoryVar)
    join_Plant_product_cost_SG3 = list_of_Plant_product_cost.reset_index().merge(list_of_Production.reset_index(), left_on=['Plant', 'Product'], right_on=['id_of_Plant', 'id_of_Product']).set_index(['id_of_Plant_product_cost', 'id_of_Plant', 'id_of_Month', 'id_of_Product'])
    reindexed_Plant_product_cost_SG3 = join_Plant_product_cost_SG3.reset_index().set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product'])
    join_Production_SG3 = list_of_Production.reset_index().merge(reindexed_Plant_product_cost_SG3.reset_index(), left_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], suffixes=('', '_right')).set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product'])
    join_Production_SG3['conditioned_Value'] = join_Production_SG3.productionVar * join_Production_SG3.Value
    agg_Production_conditioned_Value_SG3 = mdl.sum(join_Production_SG3.conditioned_Value)
    join_Marginal_profit_SG4 = list_of_Marginal_profit.reset_index().merge(list_of_ExecutedSales.reset_index(), left_on=['Market', 'Month', 'Product'], right_on=['id_of_Market', 'id_of_Month', 'id_of_Product']).set_index(['id_of_Marginal_profit', 'id_of_Market', 'id_of_Month', 'id_of_Product'])
    reindexed_Marginal_profit_SG4 = join_Marginal_profit_SG4.reset_index().set_index(['id_of_Market', 'id_of_Month', 'id_of_Product'])
    join_ExecutedSales_SG4 = list_of_ExecutedSales.reset_index().merge(reindexed_Marginal_profit_SG4.reset_index(), left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], suffixes=('', '_right')).set_index(['id_of_Market', 'id_of_Month', 'id_of_Product'])
    join_ExecutedSales_SG4['conditioned_Value'] = join_ExecutedSales_SG4.executedSalesVar * join_ExecutedSales_SG4.Value
    agg_ExecutedSales_conditioned_Value_SG4 = mdl.sum(join_ExecutedSales_SG4.conditioned_Value)
    
    kpis_expression_list = [
        (-1, 16.0, agg_ExecutedSales_executedSalesVar_SG1, 1, 0, 'market sales'),
        (1, 16.0, agg_CustomerInventory_customerInventoryVar_SG2, 1, 0, 'total inventory at markets'),
        (1, 16.0, agg_Production_conditioned_Value_SG3, 1, 0, 'overall production cost based on plant_product_cost'),
        (-1, 16.0, agg_ExecutedSales_conditioned_Value_SG4, 1, 0, 'sales profit based on marginal_profit')]
    custom_code.update_goals_list(kpis_expression_list)
    
    for _, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list:
        mdl.add_kpi(kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset), publish_name=kpi_name)
    
    mdl.minimize(sum([kpi_sign * kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset) for kpi_sign, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list]))
    
    # [ST_1] Constraint : (Guided) Executed Sales is less than the planned DEMAND_cIterativeRelationalConstraint
    # Executed Sales is less than the planned demand
    # Label: CT_1_Executed_Sales_is_less_than_the_planned_demand
    join_ExecutedSales = list_of_ExecutedSales.reset_index().merge(list_of_Demand.reset_index(), left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['Market', 'Month', 'Product']).set_index(['id_of_Market', 'id_of_Month', 'id_of_Product', 'id_of_Demand'])
    groupbyLevels = [join_ExecutedSales.index.names.index(name) for name in list_of_Demand.index.names]
    groupby_ExecutedSales = join_ExecutedSales.executedSalesVar.groupby(level=groupbyLevels).sum().to_frame()
    join_ExecutedSales_2 = groupby_ExecutedSales.join(list_of_Demand.Value, how='inner')
    for row in join_ExecutedSales_2[join_ExecutedSales_2.Value.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.executedSalesVar <= row.Value, 'Executed Sales is less than the planned demand', row)
    
    # [ST_2] Constraint : (Guided) No negative inventory at MARKET_cIterativeRelationalConstraint
    # No negative inventory at markets
    # Label: CT_2_No_negative_inventory_at_markets
    for row in list_of_CustomerInventory.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.customerInventoryVar >= 0, 'No negative inventory at markets', row)
    
    # [ST_3] Constraint : (Guided) Ensure product inventory conservation at MARKET considering sales and allocation_cIterativeRelationalConstraint
    # Ensure product inventory conservation at markets considering sales and allocation
    # Label: CT_3_Ensure_product_inventory_conservation_at_markets_considering_sales_and_allocation
    join_Delivery = list_of_Delivery.reset_index().merge(list_of_ExecutedSales.reset_index(), left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Market', 'id_of_Month', 'id_of_Product']).set_index(['id_of_Plant', 'id_of_Market', 'id_of_Month', 'id_of_Product'])
    groupbyLevels = [join_Delivery.index.names.index(name) for name in list_of_ExecutedSales.index.names]
    groupby_Delivery = join_Delivery.deliveryVar.groupby(level=groupbyLevels).sum().to_frame()
    join_CustomerInventory = list_of_CustomerInventory.reset_index().merge(list_of_ExecutedSales.reset_index(), left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Market', 'id_of_Month', 'id_of_Product']).set_index(['id_of_Market', 'id_of_Month', 'id_of_Product'])
    groupbyLevels = [join_CustomerInventory.index.names.index(name) for name in list_of_ExecutedSales.index.names]
    groupby_CustomerInventory = join_CustomerInventory.customerInventoryVar.groupby(level=groupbyLevels).sum().to_frame()
    bin_op_merge = groupby_Delivery.deliveryVar.reset_index().merge(groupby_CustomerInventory.customerInventoryVar.reset_index(), left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], how='outer').set_index(['id_of_Market', 'id_of_Month', 'id_of_Product']).fillna(0)
    bin_op_Delivery = pd.Series(bin_op_merge.deliveryVar + bin_op_merge.customerInventoryVar, name='result').to_frame()
    join_ExecutedSales = list_of_ExecutedSales.join(list_of_Month.Next, how='inner')
    join_CustomerInventory_2 = list_of_CustomerInventory.reset_index().merge(join_ExecutedSales.reset_index(), left_on=['id_of_Product', 'id_of_Market', 'id_of_Month'], right_on=['id_of_Product', 'id_of_Market', 'Next'], suffixes=('_left', '')).set_index(['id_of_Market', 'id_of_Month_left', 'id_of_Product', 'id_of_Month'])
    reindexed_CustomerInventory = join_CustomerInventory_2.reset_index().set_index(['id_of_Market', 'id_of_Product', 'id_of_Month'])
    groupbyLevels_2 = [reindexed_CustomerInventory.index.names.index(name) for name in list_of_ExecutedSales.index.names]
    groupby_CustomerInventory_2 = reindexed_CustomerInventory.customerInventoryVar.groupby(level=groupbyLevels_2).sum().to_frame()
    bin_op_merge = groupby_CustomerInventory_2.customerInventoryVar.reset_index().merge(list_of_ExecutedSales.executedSalesVar.reset_index(), left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], how='outer').set_index(['id_of_Market', 'id_of_Product', 'id_of_Month']).fillna(0)
    bin_op_CustomerInventory = pd.Series(bin_op_merge.customerInventoryVar + bin_op_merge.executedSalesVar, name='result').to_frame()
    join_Delivery_2 = bin_op_Delivery.reset_index().merge(bin_op_CustomerInventory.reset_index()[['id_of_Market', 'id_of_Product', 'id_of_Month', 'result']], left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], suffixes=('', '_right'), how='inner').set_index(['id_of_Market', 'id_of_Month', 'id_of_Product'])
    for row in join_Delivery_2.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.result == row.result_right, 'Ensure product inventory conservation at markets considering sales and allocation', row)
    
    # [ST_4] Constraint : (Guided) Delivery at MARKET comes from production and inventory at PLANT_cIterativeRelationalConstraint
    # Delivery at markets comes from production and inventory at plants
    # Label: CT_4_Delivery_at_markets_comes_from_production_and_inventory_at_plants
    join_Delivery = list_of_Delivery.reset_index().merge(list_of_Production.reset_index(), left_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Plant', 'id_of_Month', 'id_of_Product']).set_index(['id_of_Plant', 'id_of_Market', 'id_of_Month', 'id_of_Product'])
    groupbyLevels = [join_Delivery.index.names.index(name) for name in list_of_Production.index.names]
    groupby_Delivery = join_Delivery.deliveryVar.groupby(level=groupbyLevels).sum().to_frame()
    join_Inventory = list_of_Inventory.reset_index().merge(list_of_Production.reset_index(), left_on=['id_of_Product', 'id_of_Plant', 'id_of_Month'], right_on=['id_of_Product', 'id_of_Plant', 'id_of_Month']).set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product'])
    groupbyLevels = [join_Inventory.index.names.index(name) for name in list_of_Production.index.names]
    groupby_Inventory = join_Inventory.inventoryVar.groupby(level=groupbyLevels).sum().to_frame()
    join_Production = list_of_Production.join(list_of_Month.Next, how='inner')
    join_Inventory_2 = list_of_Inventory.reset_index().merge(join_Production.reset_index(), left_on=['id_of_Product', 'id_of_Plant', 'id_of_Month'], right_on=['id_of_Product', 'id_of_Plant', 'Next'], suffixes=('_left', '')).set_index(['id_of_Plant', 'id_of_Month_left', 'id_of_Product', 'id_of_Month'])
    reindexed_Inventory = join_Inventory_2.reset_index().set_index(['id_of_Plant', 'id_of_Product', 'id_of_Month'])
    groupbyLevels_2 = [reindexed_Inventory.index.names.index(name) for name in list_of_Production.index.names]
    groupby_Inventory_2 = reindexed_Inventory.inventoryVar.groupby(level=groupbyLevels_2).sum().to_frame()
    bin_op_merge = groupby_Inventory.inventoryVar.reset_index().merge(groupby_Inventory_2.inventoryVar.reset_index(), left_on=['id_of_Plant', 'id_of_Product', 'id_of_Month'], right_on=['id_of_Plant', 'id_of_Product', 'id_of_Month'], suffixes=('', '_right'), how='outer').set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product']).fillna(0)
    bin_op_Inventory = pd.Series(bin_op_merge.inventoryVar - bin_op_merge.inventoryVar_right, name='result').to_frame()
    bin_op_merge_2 = bin_op_Inventory.result.reset_index().merge(list_of_Production.productionVar.reset_index(), left_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], how='outer').set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product']).fillna(0)
    bin_op_Inventory_2 = pd.Series(bin_op_merge_2.result + bin_op_merge_2.productionVar, name='result').to_frame()
    join_Delivery_2 = groupby_Delivery.reset_index().merge(bin_op_Inventory_2.reset_index()[['id_of_Plant', 'id_of_Month', 'id_of_Product', 'result']], left_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], how='inner').set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product'])
    for row in join_Delivery_2.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.deliveryVar == row.result, 'Delivery at markets comes from production and inventory at plants', row)
    
    # [ST_5] Constraint : (Guided) Deliver at least firm_sales to market_cIterativeRelationalConstraint
    # Deliver at least firm_sales to markets
    # Label: CT_5_Deliver_at_least_firm_sales_to_markets
    join_Delivery = list_of_Delivery.reset_index().merge(list_of_Firm_sales.reset_index(), left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['Market', 'Month', 'Product']).set_index(['id_of_Plant', 'id_of_Market', 'id_of_Month', 'id_of_Product', 'id_of_Firm_sales'])
    groupbyLevels = [join_Delivery.index.names.index(name) for name in list_of_Firm_sales.index.names]
    groupby_Delivery = join_Delivery.deliveryVar.groupby(level=groupbyLevels).sum().to_frame()
    join_Delivery_2 = groupby_Delivery.join(list_of_Firm_sales.Value, how='inner')
    for row in join_Delivery_2[join_Delivery_2.Value.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.deliveryVar >= row.Value, 'Deliver at least firm_sales to markets', row)
    
    # [ST_6] Constraint : (Guided) For each PLANT_MONTH_CAPACITY, total production where production is joined to PLANT_MONTH_CAPACITY is less than or equal to Value of PLANT_MONTH_CAPACITY_cIterativeRelationalConstraint
    # For each plant_month_capacity, total Productions where Production is joined to plant_month_capacity is less than or equal to Value
    # Label: CT_6_For_each_plant_month_capacity__total_Productions_where_Production_is_joined_to_plant_month_capacity_is_less_than_or_equal_to_Value
    join_Production = list_of_Production.reset_index().merge(list_of_Plant_month_capacity.reset_index(), left_on=['id_of_Plant', 'id_of_Month'], right_on=['Plant', 'Month']).set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product', 'id_of_Plant_month_capacity'])
    groupbyLevels = [join_Production.index.names.index(name) for name in list_of_Plant_month_capacity.index.names]
    groupby_Production = join_Production.productionVar.groupby(level=groupbyLevels).sum().to_frame()
    join_Production_2 = groupby_Production.join(list_of_Plant_month_capacity.Value, how='inner')
    for row in join_Production_2[join_Production_2.Value.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.productionVar <= row.Value, 'For each plant_month_capacity, total Productions where Production is joined to plant_month_capacity is less than or equal to Value', row)
    
    # [ST_7] Constraint : (Guided) For each production, total production is less than or equal to total Value of PLANT_PRODUCT_CAPACITY where PLANT_PRODUCT_CAPACITY is joined to production_cIterativeRelationalConstraint
    # For each Production, Production is less than or equal to total plant_product_capacities where plant_product_capacity is joined to Production
    # Label: CT_7_For_each_Production__Production_is_less_than_or_equal_to_total_plant_product_capacities_where_plant_product_capacity_is_joined_to_Production
    join_Plant_product_capacity = list_of_Plant_product_capacity.reset_index().merge(list_of_Production.reset_index(), left_on=['Plant', 'Product'], right_on=['id_of_Plant', 'id_of_Product']).set_index(['id_of_Plant_product_capacity', 'id_of_Plant', 'id_of_Month', 'id_of_Product'])
    reindexed_Plant_product_capacity = join_Plant_product_capacity.reset_index().set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product'])
    join_Production = list_of_Production.reset_index().merge(reindexed_Plant_product_capacity.reset_index()[['id_of_Plant', 'id_of_Month', 'id_of_Product', 'id_of_Plant_product_capacity']], left_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], how='inner').set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product'])
    for row in join_Production[join_Production.id_of_Plant_product_capacity.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.productionVar <= row.id_of_Plant_product_capacity, 'For each Production, Production is less than or equal to total plant_product_capacities where plant_product_capacity is joined to Production', row)
    
    # [ST_8] Constraint : (Guided) For each production, total delivery where delivery is joined to production is less than or equal to production_cIterativeRelationalConstraint
    # For each Production, total Deliveries where Delivery is joined to Production is less than or equal to Production
    # Label: CT_8_For_each_Production__total_Deliveries_where_Delivery_is_joined_to_Production_is_less_than_or_equal_to_Production
    join_Delivery = list_of_Delivery.reset_index().merge(list_of_Production.reset_index(), left_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Plant', 'id_of_Month', 'id_of_Product']).set_index(['id_of_Plant', 'id_of_Market', 'id_of_Month', 'id_of_Product'])
    groupbyLevels = [join_Delivery.index.names.index(name) for name in list_of_Production.index.names]
    groupby_Delivery = join_Delivery.deliveryVar.groupby(level=groupbyLevels).sum().to_frame()
    join_Delivery_2 = groupby_Delivery.reset_index().merge(list_of_Production.reset_index()[['id_of_Plant', 'id_of_Month', 'id_of_Product', 'productionVar']], left_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Plant', 'id_of_Month', 'id_of_Product'], how='inner').set_index(['id_of_Plant', 'id_of_Month', 'id_of_Product'])
    for row in join_Delivery_2.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.deliveryVar <= row.productionVar, 'For each Production, total Deliveries where Delivery is joined to Production is less than or equal to Production', row)
    
    # [ST_9] Constraint : (Guided) Set initial market inventory for first_month of parameters to initial_inventory_cIterativeRelationalConstraint
    # Set initial market inventory for first_month of parameters to initial_inventories
    # Label: CT_9_Set_initial_market_inventory_for_first_month_of_parameters_to_initial_inventories
    join_CustomerInventory = list_of_CustomerInventory.reset_index().merge(list_of_Parameters.reset_index(), left_on=['id_of_Month'], right_on=['first_month']).set_index(['id_of_Market', 'id_of_Month', 'id_of_Product', 'id_of_Parameters'])
    join_Initial_inventory = list_of_Initial_inventory.reset_index().merge(list_of_CustomerInventory.reset_index(), left_on=['Market', 'Product'], right_on=['id_of_Market', 'id_of_Product']).set_index(['id_of_Initial_inventory', 'id_of_Market', 'id_of_Month', 'id_of_Product'])
    join_CustomerInventory_2 = join_CustomerInventory.reset_index().merge(join_Initial_inventory.reset_index()[['id_of_Initial_inventory', 'id_of_Market', 'id_of_Month', 'id_of_Product', 'Value']], left_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], right_on=['id_of_Market', 'id_of_Month', 'id_of_Product'], how='inner').set_index(['id_of_Market', 'id_of_Month', 'id_of_Product', 'id_of_Parameters', 'id_of_Initial_inventory'])
    for row in join_CustomerInventory_2[join_CustomerInventory_2.Value.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.customerInventoryVar == row.Value, 'Set initial market inventory for first_month of parameters to initial_inventories', row)
    
    # [ST_10] Constraint : (Guided) No stock at plant_cIterativeRelationalConstraint
    # No stock at plants
    # Label: CT_10_No_stock_at_plants
    for row in list_of_Inventory.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.inventoryVar == 0, 'No stock at plants', row)


    return mdl


def solve_model(mdl):
    mdl.parameters.timelimit = 120
    # Call to custom code to update parameters value
    custom_code.update_solver_params(mdl.parameters)
    # Update parameters value based on environment variables definition
    cplex_param_env_prefix = 'ma.cplex.'
    cplex_params = [name.qualified_name for name in mdl.parameters.generate_params()]
    for param in cplex_params:
        env_param = cplex_param_env_prefix + param
        param_value = get_environment().get_parameter(env_param)
        if param_value:
            # Updating parameter value
            print("Updated value for parameter %s = %s" % (param, param_value))
            parameters = mdl.parameters
            for p in param.split('.')[1:]:
                parameters = parameters.__getattribute__(p)
            parameters.set(param_value)

    msol = mdl.solve(log_output=True)
    if not msol:
        print("!!! Solve of the model fails")
        if mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_SOLUTION or mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION:
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
    list_of_Production_solution = pd.DataFrame(index=list_of_Production.index)
    list_of_Production_solution['productionVar'] = msol.get_values(list_of_Production.productionVar.values)
    list_of_ExecutedSales_solution = pd.DataFrame(index=list_of_ExecutedSales.index)
    list_of_ExecutedSales_solution['executedSalesVar'] = msol.get_values(list_of_ExecutedSales.executedSalesVar.values)
    list_of_CustomerInventory_solution = pd.DataFrame(index=list_of_CustomerInventory.index)
    list_of_CustomerInventory_solution['customerInventoryVar'] = msol.get_values(list_of_CustomerInventory.customerInventoryVar.values)
    list_of_Inventory_solution = pd.DataFrame(index=list_of_Inventory.index)
    list_of_Inventory_solution['inventoryVar'] = msol.get_values(list_of_Inventory.inventoryVar.values)
    list_of_Delivery_solution = pd.DataFrame(index=list_of_Delivery.index)
    list_of_Delivery_solution['deliveryVar'] = msol.get_values(list_of_Delivery.deliveryVar.values)

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_Production_solution'] = list_of_Production_solution.reset_index()
        outputs['list_of_Delivery_solution'] = list_of_Delivery_solution.reset_index()
        outputs['list_of_ExecutedSales_solution'] = list_of_ExecutedSales_solution.reset_index()
        outputs['list_of_Inventory_solution'] = list_of_Inventory_solution.reset_index()
        outputs['list_of_CustomerInventory_solution'] = list_of_CustomerInventory_solution.reset_index()
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
                                       'update_goals_list': (lambda *args: None),
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
msol = solve_model(model)
elapsed_time = time.time() - start_time
print('model solve done in ' + str(elapsed_time) + ' secs')
if msol:
    export_solution(msol)
