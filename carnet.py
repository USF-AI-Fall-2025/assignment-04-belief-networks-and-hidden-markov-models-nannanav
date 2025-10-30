from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

def setup():
    car_model = DiscreteBayesianNetwork(
        [
            ("Battery", "Radio"),
            ("Battery", "Ignition"),
            ("Ignition","Starts"),
            ("Gas","Starts"),
            ("Starts","Moves"),
    ])

    # Defining the parameters using CPD

    cpd_battery = TabularCPD(
        variable="Battery", variable_card=2, values=[[0.70], [0.30]],
        state_names={"Battery":['Works',"Doesn't work"]},
    )

    cpd_gas = TabularCPD(
        variable="Gas", variable_card=2, values=[[0.40], [0.60]],
        state_names={"Gas":['Full',"Empty"]},
    )

    cpd_radio = TabularCPD(
        variable=  "Radio", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Radio": ["turns on", "Doesn't turn on"],
                    "Battery": ['Works',"Doesn't work"]}
    )

    cpd_ignition = TabularCPD(
        variable=  "Ignition", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Ignition": ["Works", "Doesn't work"],
                    "Battery": ['Works',"Doesn't work"]}
    )

    cpd_starts = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
        evidence=["Ignition", "Gas"],
        evidence_card=[2, 2],
        state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
    )

    cpd_moves = TabularCPD(
        variable="Moves", variable_card=2,
        values=[[0.8, 0.01],[0.2, 0.99]],
        evidence=["Starts"],
        evidence_card=[2],
        state_names={"Moves": ["yes", "no"],
                    "Starts": ['yes', 'no'] }
    )


    # Associating the parameters with the model structure
    car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

    car_infer = VariableElimination(car_model)
    return car_model, car_infer, cpd_starts

def execute_queries(car_infer):
    print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))
    print()

    #Given that the car will not move, what is the probability that the battery is not working?
    print(car_infer.query(variables=["Battery"],evidence={"Moves":"no"}))
    print()

    #Given that the radio is not working, what is the probability that the car will not start?
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"}))
    print()

    #Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works"}))
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works", "Gas":"Full"}))
    print("From the car_model above, we know that the radio only depends on battery. So, a change/discovery in gas does not affect the radio")
    print()

    #Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it?
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no"}))
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no", "Gas":"Empty"}))
    print("Moves depends on starts, which depends on ignition and gas. Given that moves=no, if gas is discovered to be empty, then there's a lower chance that its because of ignition not happening")
    print()

    #What is the probability that the car starts if the radio works and it has gas in it? Include each of your queries in carnet.py. Also, please add a main that executes your queries
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"turns on", "Gas":"Full"}))
    print()

def setup2(car_model, cpd_to_remove):
    car_model.add_edge("KeyPresent", "Starts")
    cpd_key_present = TabularCPD(
        variable="KeyPresent", variable_card=2, values=[[0.70], [0.30]],
        state_names={"KeyPresent":["yes","no"]},
    )
    cpd_starts = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
        evidence=["Ignition", "Gas", "KeyPresent"],
        evidence_card=[2, 2, 2],
        state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent":["yes","no"]},
    )
    car_model.remove_cpds(cpd_to_remove)
    car_model.add_cpds(cpd_key_present, cpd_starts)
    car_infer = VariableElimination(car_model)
    return car_infer


# P(starts | gas, ignition, keyPresent) = 0.99
# P(starts | gas, !ignition, keyPresent) = 0.01
# P(starts | !gas, ignition, keyPresent) = 0.01
# P(starts | !gas, !ignition, keyPresent) = 0.01
# P(starts | gas, ignition, !keyPresent) = 0.01
# P(starts | gas, !ignition, !keyPresent) = 0.01
# P(starts | !gas, ignition, !keyPresent) = 0.01
# P(starts | !gas, !ignition, !keyPresent) = 0.01

def execute_queries2(car_infer):
    #Add a query showing the probability that the key is not present given that the car does not move.
    print(car_infer.query(variables=["KeyPresent"],evidence={"Moves":"no"}))
    print()

if __name__ == "__main__":
    car_model, car_infer, cpd_to_remove = setup()
    execute_queries(car_infer)

    car_infer = setup2(car_model, cpd_to_remove)
    execute_queries2(car_infer)
