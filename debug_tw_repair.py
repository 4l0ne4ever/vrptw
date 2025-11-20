"""
Script để phân tích complexity và performance của TW repair.
Chạy script này để xem thực tế TW repair mất bao nhiêu thời gian.
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.algorithms.decoder import RouteDecoder
from src.algorithms.tw_repair import TWRepairOperator
from app.services.data_service import DataService

def analyze_tw_repair_performance():
    """Phân tích performance của TW repair."""
    
    # Load một dataset mẫu
    data_service = DataService()
    
    # Test với Solomon dataset
    print("Loading Solomon dataset...")
    try:
        solomon_dir = Path("data/datasets/solomon")
        test_file = list(solomon_dir.glob("C101.json"))[0] if list(solomon_dir.glob("C101.json")) else None
        
        if not test_file:
            print("No test dataset found")
            return

        with open(test_file, 'r') as f:
            data_dict = json.load(f)
        
        problem = data_service.create_vrp_problem(data_dict, dataset_type="solomon")
        
        # Tạo một chromosome mẫu
        from src.algorithms.genetic_algorithm import GeneticAlgorithm
        from config import GA_CONFIG
        
        ga = GeneticAlgorithm(problem, GA_CONFIG)
        ga.initialize_population()
        
        # Test với 10 individuals đầu tiên
        decoder = RouteDecoder(problem, use_split_algorithm=True)
        tw_repair = TWRepairOperator(problem, max_iterations=50)
        
        total_decode_time = 0.0
        total_repair_time = 0.0
        repair_calls = 0
        repair_skipped = 0
        
        print(f"\nTesting with {len(ga.population.individuals)} individuals...")
        
        for i, ind in enumerate(ga.population.individuals[:10]):
            # Decode
            decode_start = time.perf_counter()
            routes = decoder.decode_chromosome(ind.chromosome)
            decode_time = time.perf_counter() - decode_start
            total_decode_time += decode_time
            
            # Check if repair would be called
            total_lateness = sum(
                tw_repair._route_lateness(route) for route in routes
            )
            
            if total_lateness > 1e-6:
                repair_start = time.perf_counter()
                repaired = tw_repair.repair_routes(routes)
                repair_time = time.perf_counter() - repair_start
                total_repair_time += repair_time
                repair_calls += 1
                print(f"  Individual {i}: decode={decode_time*1000:.1f}ms, "
                      f"repair={repair_time*1000:.1f}ms, "
                      f"lateness={total_lateness:.2f}, routes={len(routes)}")
            else:
                repair_skipped += 1
                print(f"  Individual {i}: decode={decode_time*1000:.1f}ms, "
                      f"repair=SKIPPED (no violations), routes={len(routes)}")
        
        print(f"\n=== Summary ===")
        print(f"Total decode time: {total_decode_time*1000:.1f}ms")
        print(f"Total repair time: {total_repair_time*1000:.1f}ms")
        print(f"Repair calls: {repair_calls}")
        print(f"Repair skipped: {repair_skipped}")
        print(f"Average decode time: {total_decode_time*1000/10:.1f}ms")
        if repair_calls > 0:
            print(f"Average repair time: {total_repair_time*1000/repair_calls:.1f}ms")
        print(f"\nRepair overhead: {total_repair_time/total_decode_time*100:.1f}% of decode time")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    analyze_tw_repair_performance()

