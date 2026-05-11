import sys
sys.path.insert(0, 'scripts')
from visualize import Visualizer
from metrics import MetricsProcessor

processor = MetricsProcessor()
processor.load_from_results()

viz = Visualizer(metrics_processor=processor)
viz.generate_all_visualizations()

print("Done! Check plots/figures/ and plots/tables/")
