import dac
import inspect

model = dac.utils.load_model(
    model_type="24khz",
    model_bitrate="8kbps"
)

print(inspect.signature(model.encode))