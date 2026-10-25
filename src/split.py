import splitfolders

input_folder = "data/PetImages" 
output_folder = "data/splitted-data" 

splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42,           
    ratio=(0.8, 0.2),  
    group_prefix=None, 
    move=False         
)