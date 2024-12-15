from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Ranker:
    @staticmethod
    def rank(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Ranking results...")
        return sorted(results, key=lambda x: x['distance'])

def test_ranker():
    ranker = Ranker()
    test_results = [
        {'id': '1', 'document': 'Doc1', 'distance': 0.5},
        {'id': '2', 'document': 'Doc2', 'distance': 0.3},
        {'id': '3', 'document': 'Doc3', 'distance': 0.7},
    ]
    ranked_results = ranker.rank(test_results)
    logger.info("Ranked results:")
    for result in ranked_results:
        logger.info(result)

if __name__ == "__main__":
    test_ranker()