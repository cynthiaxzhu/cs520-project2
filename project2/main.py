import bots
from ship import Spaceship


def run_experiments():
    bot_num_func = {
        1: bots.bot_1,
        2: bots.bot_2,
        3: bots.bot_3,
        4: bots.bot_4,
        5: bots.bot_5,
        6: bots.bot_6,
        7: bots.bot_7,
        8: bots.bot_8
    }
    
    bot_num_num_aliens = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2
    }
    
    grid_size = 20
    
    # apothems = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    apothems = [2]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_trials = 100
    
    results = []
    
    for bot_num in range(1, 9):
        results_bot = []
        
        for apothem in apothems:
            avg_num_saved_list = []
            avg_num_time_steps_list = []
            success_rate_list = []
            
            for alpha in alphas:
                num_saved_list = []
                num_time_steps_list = []
                is_success_list = []
                
                for _ in range(num_trials):
                    spaceship = Spaceship(
                        grid_size = grid_size,
                        num_aliens = bot_num_num_aliens[bot_num]
                    )
                    
                    num_saved, num_time_steps, is_success = (
                        bot_num_func[bot_num](
                            spaceship = spaceship,
                            apothem = apothem,
                            alpha = alpha
                        )
                    )
                    
                    num_saved_list.append(num_saved)
                    num_time_steps_list.append(num_time_steps)
                    is_success_list.append(is_success)
                
                avg_num_saved = sum(num_saved_list) / num_trials
                avg_num_time_steps = sum(num_time_steps_list) / num_trials
                success_rate = sum(is_success_list) / num_trials
                
                avg_num_saved_list.append(avg_num_saved)
                avg_num_time_steps_list.append(avg_num_time_steps)
                success_rate_list.append(success_rate)
            
            results_apothem = [
                avg_num_saved_list,
                avg_num_time_steps_list,
                success_rate_list
            ]
            
            results_bot.append(results_apothem)
        
        results.append(results_bot)
    
    return results


def main():
    # results = run_experiments()
    
    # testing
    spaceship = Spaceship(grid_size = 4, num_aliens = 2)
    
    print(spaceship.grid)
    print("number of open cells:", spaceship.num_open_cells)
    print()
    
    num_saved, num_time_steps, is_success = (
        bots.bot_7(spaceship = spaceship, apothem = 1, alpha = 0.1)
    )
    
    print("num_saved:", num_saved)
    print("num_time_steps:", num_time_steps)
    print("is_success:", is_success)


if __name__ == "__main__":
    main()
