from utils.bisp_instance_generator import InstanceGenerator

def generate_instances(number_of_instances, params):
    instance_generator = InstanceGenerator()
    instance_generator.generate_instance(number_of_instances, params)

def generate_example_instance():
    instance_generator = InstanceGenerator()
    instance_generator.generate_example_instance()

if __name__ == "__main__":
    params = {
        "nL": [5],
        "nF": [5],
        "p": [.25, .50, .75],
        "rhs_ratio": [-.1, 0, .1],
    }

    generate_instances(10, params)