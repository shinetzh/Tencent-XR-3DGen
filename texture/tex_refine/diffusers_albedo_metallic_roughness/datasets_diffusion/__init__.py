from datasets_diffusion.datasets_mod_transfer_metallic import DatasetModTransfer_metallic
from datasets_diffusion.datasets_mod_transfer_metallic_roughness_gray_combine import DatasetModTransfer_metallic_roughness
from datasets_diffusion.datasets_mod_transfer_metallic_roughness_gray_combine_6views_two_outputs_h20 import DatasetModTransfer_metallic_roughness_two_outputs_h20
from datasets_diffusion.datasets_mod_transfer_metallic_roughness_gray_combine_6views_two_outputs import DatasetModTransfer_metallic_roughness_two_outputs
from datasets_diffusion.datasets_mod_transfer_metallic_roughness_gray_combine_6views_two_outputs_3channels import DatasetModTransfer_metallic_roughness_two_outputs_3channels
from datasets_diffusion.datasets_mod_transfer_albedo_gray import DatasetModTransfer_albedo

from datasets_diffusion.datasets_mod_transfer_delight_single_albedo import DatasetModTransfer_single_albedo
from datasets_diffusion.datasets_mod_transfer_delight_single_full_light import DatasetModTransfer_single_full_light

from datasets_diffusion.datasets_mod_transfer_albedo_gray_6views_1024_bright import DatasetModTransfer_albedo_6views_1024
from datasets_diffusion.datasets_mod_transfer_albedo_gray_6views_512_bright import DatasetModTransfer_albedo_6views_512

from datasets_diffusion.datasets_mod_transfer_albedo_metallic_roughness_gray_combine_6views import DatasetModTransfer_albedo_metallic_roughness_6views

def get_dataset(configs, data_type="train", resample=True, load_from_cache_last=False):
    if configs["data_config"]["dataset_name"] == "DatasetModTransfer_albedo_metallic_roughness_6views":
        return DatasetModTransfer_albedo_metallic_roughness_6views(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_metallic":
        return DatasetModTransfer_metallic(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_metallic_roughness":
        return DatasetModTransfer_metallic_roughness(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_metallic_roughness_two_outputs":
        return DatasetModTransfer_metallic_roughness_two_outputs(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_metallic_roughness_two_outputs_3channels":
        return DatasetModTransfer_metallic_roughness_two_outputs_3channels(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_albedo":
        return DatasetModTransfer_albedo(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_albedo_6views_1024":
        return DatasetModTransfer_albedo_6views_1024(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_albedo_6views_512":
        return DatasetModTransfer_albedo_6views_512(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_single_albedo":
        return DatasetModTransfer_single_albedo(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "DatasetModTransfer_single_full_light":
        return DatasetModTransfer_single_full_light(configs, data_type=data_type)


        
