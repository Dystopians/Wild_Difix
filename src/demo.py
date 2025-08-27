from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")
input_image = load_image("/data2/peilincai/Difix3D/assets/ghosts/route_pt001_h135_p00_fov90.png")
ref_image = load_image("/data2/peilincai/Difix3D/assets/images/route_pt001_h090_p00_fov90.jpg")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("route_e_example_output.png")