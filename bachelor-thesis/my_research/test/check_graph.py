from utils.config_loader import load_config
from evaluation.utils_eval import build_graph_from_topology

def main():
    links = config["topology"]["links"]
    graph = build_graph_from_topology(links)

    print(graph.nodes())
    print(graph.edges(data=True))


if __name__=="__main__":
    main()