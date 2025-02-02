# Copyright 2024 Nimrod Curtis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard library imports
import yaml

# Custom libraries
from bagtool.process.process import BagProcess as bp

def main():
    
    # Set the config path
    config_path = 'process_bag_config.yaml'
    with open(config_path, 'r') as file:
            process_config = yaml.safe_load(file)
    
    # Process based on the config
    bp.process_folder(config=process_config)

if __name__ == "__main__":
    main()