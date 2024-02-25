from datetime import datetime

from utils.instance_generator import InstanceGenerator

def generate_instances(instance_type, number_of_instances, params):
    instance_generator = InstanceGenerator()
    instance_generator.generate_instance(instance_type, number_of_instances, params)

if __name__ == "__main__":
    params = {
        "n_L": 50,
        "n_F": 50, 
        "m_L": 10,
        "m_F": 10,
        "folder_name": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    }
    generate_instances("weak_leader", 10, params)