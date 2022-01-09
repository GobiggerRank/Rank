import argparse
import importlib
import logging
import time
import os
import sys
import json
import pickle
import trueskill
import numpy as np
from pathlib import Path
from typing import List

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

use_cpp = True
team_num = 4
match_time = 600

result_folder = Path("result")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=True, action='store_true')
    parser.add_argument('--battle', default=True, action='store_true')
    parser.add_argument('--submit-folder', type=str, default="GoBiggerSubmit/submit")
    parser.add_argument('--save-result', type=str, default="")
    parser.add_argument('--each-battle-size', type=int, default=20)
    parser.add_argument('--battle-count', type=int, default=30)
    parser.add_argument('--use-single-thread', default=False, action='store_true')
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_known_args()[0]
    return args


def battle_single(submits: List[str], cfg, seed: int = 0):
    if use_cpp:
        sys.path.append(str(Path(Path(__file__).resolve().parent, "python_lib")))
        from gobigger_cpp import Server
        server = Server(cfg)
    else:
        from gobigger.server import Server
        from gobigger.render import EnvRender
        server = Server(cfg)
        render = EnvRender(server.map_width, server.map_height)
        server.set_render(render)
    assert len(submits) == server.team_num
    start_time = time.time()
    server.seed(seed)
    server.start()

    agents = []
    team_player_names = server.get_team_names()
    team_names = list(team_player_names.keys())
    args = get_args()
    for index, submit in enumerate(submits):
        try:
            p = importlib.import_module(f"{submit}.my_submission")
            agents.append(p.MySubmission(team_name=team_names[index],
                                         player_names=team_player_names[team_names[index]]))
        except Exception as e:
            print(f"You must implement `MySubmission` in {submit} my_submission.py ! exception: {e}")
            # raise

    time_obs = 0
    time_step = 0
    time_actions = [0 for _ in range(len(team_player_names))]
    all_tick = match_time * 5
    try:
        for tick in range(all_tick + 10):
            t = time.time()
            obs = server.obs()
            tmp_obs = time.time() - t
            time_obs += tmp_obs

            t = time.time()
            global_state, player_states = obs
            actions = {}
            for index, agent in enumerate(agents):
                agent_obs = [global_state, {
                    player_name: player_states[player_name] for player_name in agent.player_names
                }]
                t1 = time.time()
                # TODO check action
                action = agent.get_actions(obs=agent_obs)
                time_actions[index] += time.time() - t1
                actions.update(action)
            time_action = time.time() - t
            t = time.time()
            finish_flag = server.step(actions=actions)
            tmp_step = time.time() - t
            time_step += tmp_step

            leaderboard = global_state["leaderboard"].values()
            leaderboard = [f"{_:.1f}" for _ in leaderboard]

            logging.debug(
                f"tick: {tick:>4}/{all_tick} obs: {tmp_obs:.3f} step: {tmp_step:.3f} action: {time_action:.3f} "
                f"score: {leaderboard}")

            if finish_flag:
                logging.debug('Game Over')
                break

        leaderboard = server.obs()[0]['leaderboard']
        server.close()
        scores = list(leaderboard.values())
    except Exception as e:
        scores = [0] * team_num
        print(f"battle error: {e}")
    s = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

    ranks = []
    for t in range(team_num):
        ranks.append(s.index(t))
    data = {
        "seed": seed,
        "submits": submits,
        "ranks": ranks,
        "scores": scores,
        "time_actions": time_actions,
        "time_cost": time.time() - start_time,
    }
    return data


def battle_vs(submits_vec: List[List[str]]):
    cfg = dict(
        team_num=team_num,
        match_time=match_time,
        obs_settings=dict(
            with_spatial=False,
        ),
    )
    start_time = time.time()

    logging.disable()

    datas = []

    state = np.random.RandomState(int(time.time()))
    args = get_args()
    if args.use_single_thread:
        for submits in submits_vec:
            logging.disable()
            data = battle_single(submits, cfg, seed=state.randint(0, 1e8))
            datas.append(data)
            logging.disable(logging.NOTSET)
            logging.debug(f"{data}")
    else:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=args.max_workers)
        futures = []
        for submits in submits_vec:
            future = executor.submit(battle_single, submits, cfg, seed=state.randint(0, 1e8))
            futures.append(future)

        logging.disable(logging.NOTSET)
        for f in futures:
            data = f.result()
            datas.append(data)
            logging.debug(f"{data}")

    logging.debug(f"all time cost: {time.time() - start_time:.2f}s")
    return datas


def find_submit(submit_folder):
    submits = []
    args = get_args()
    for (dir_path, dir_names, file_names) in os.walk(submit_folder):
        if "__pycache__" in dir_path:
            continue
        submit = str(dir_path).replace("/", ".")
        try:
            p = importlib.import_module(f"{submit}.my_submission")
            p.MySubmission(team_name="0", player_names=["0", "1", "2"])
            submits.append(submit)
        except Exception as e:
            logging.warning(f"submit: [{submit}] error, exception: {e}")
            continue
    return submits


def get_all_datas():
    all_datas_map = {}
    results_files = []
    for (dir_path, _, file_names) in os.walk(result_folder):
        for file_name in file_names:
            if str(file_name).endswith("json"):
                results_files.append(Path(dir_path, file_name))
    for results_file in results_files:
        with open(results_file, "r") as f:
            all_datas_map[str(results_file)] = json.load(f)
    return all_datas_map


def init_submit_count(submits, all_datas_map):
    submit_count = {submit: 0 for submit in submits}
    for json_file in all_datas_map:
        datas = all_datas_map[json_file]
        for data in datas:
            if "submits" not in data:
                continue
            for s in data["submits"]:
                if s not in submit_count:
                    continue
                submit_count[s] += 1
    logging.info(f"submit_count: {submit_count}")
    return submit_count


def collect_new_datas(all_datas_map, submit_count, args):
    counts = np.array(list(submit_count.values()))
    counts = np.max(counts) + 10 - counts
    proba = counts / np.sum(counts)
    p = {k: proba[index] for index, k in enumerate(submit_count.keys())}
    logging.info(f"submit proba:\n{json.dumps(p, indent=4)}")

    submits_vec = []
    for _ in range(int(args.each_battle_size)):
        choice = list(np.random.choice(list(submit_count.keys()), size=team_num, replace=False, p=proba))
        submits_vec.append(choice)
        for c in choice:
            submit_count[c] += 1
    datas = battle_vs(submits_vec)
    local_time = time.localtime()
    date = time.strftime("%Y-%m-%d", local_time)
    date_folder = Path(result_folder, date)
    date_folder.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%H_%M_%S", local_time)
    data_file = Path(date_folder, f"{timestamp}_{int(time.time() * 1000) % 1000:03d}.json")
    with open(data_file, "w") as f:
        json.dump(datas, f, indent=4)
    all_datas_map[data_file] = datas
    return {data_file: datas}


def show_table(records, fields, headings, alignment=None):
    left_rule = {'<': ':', '^': ':', '>': '-'}
    right_rule = {'<': '-', '^': ':', '>': ':'}

    def evaluate_field(record, field_spec):
        if type(field_spec) is int:
            return str(record[field_spec])
        elif type(field_spec) is str:
            return str(getattr(record, field_spec))
        else:
            return str(field_spec(record))

    num_columns = len(fields)
    assert len(headings) == num_columns

    # Compute the table cell data
    columns = [[] for _ in range(num_columns)]
    for record in records:
        for i, field in enumerate(fields):
            columns[i].append(evaluate_field(record, field))

    extended_align = alignment if alignment is not None else []
    if len(extended_align) > num_columns:
        extended_align = extended_align[0:num_columns]
    elif len(extended_align) < num_columns:
        extended_align += [('^', '<') for _ in range(num_columns - len(extended_align))]

    heading_align, cell_align = [x for x in zip(*extended_align)]

    field_widths = [len(max(column, key=len)) if len(column) > 0 else 0 for column in columns]
    heading_widths = [max(len(head), 2) for head in headings]
    column_widths = [max(x) for x in zip(field_widths, heading_widths)]

    _ = ' | '.join(['{:' + a + str(w) + '}' for a, w in zip(heading_align, column_widths)])
    heading_template = '| ' + _ + ' |'
    _ = ' | '.join(['{:' + a + str(w) + '}' for a, w in zip(cell_align, column_widths)])
    row_template = '| ' + _ + ' |'

    _ = ' | '.join([left_rule[a] + '-' * (w - 2) + right_rule[a] for a, w in zip(cell_align, column_widths)])
    ruling = '| ' + _ + ' |'

    ret = ""
    ret += heading_template.format(*headings).rstrip() + '\n'
    ret += ruling.rstrip() + '\n'
    for row in zip(*columns):
        ret += row_template.format(*row).rstrip() + '\n'
    return ret


def rank(submit_rating, all_datas_map, env, json_files_set, submit_count):
    start_time = time.time()
    for json_file in all_datas_map:
        if json_file in json_files_set:
            continue
        json_files_set.add(json_file)
        datas = all_datas_map[json_file]
        for data in datas:
            flag = False
            for s in data["submits"]:
                if s not in submit_rating:
                    flag = True
            if flag:
                continue
            r0 = submit_rating[data["submits"][0]]
            r1 = submit_rating[data["submits"][1]]
            r2 = submit_rating[data["submits"][2]]
            r3 = submit_rating[data["submits"][3]]
            rating_groups = [(r0,), (r1,), (r2,), (r3,)]
            rated_rating_groups = env.rate(rating_groups, ranks=data["ranks"])
            (r0,), (r1,), (r2,), (r3,) = rated_rating_groups
            submit_rating[data["submits"][0]] = r0
            submit_rating[data["submits"][1]] = r1
            submit_rating[data["submits"][2]] = r2
            submit_rating[data["submits"][3]] = r3
    submit_rating_vec = []
    args = get_args()
    for s, r in submit_rating.items():
        submit_rating_vec.append([s.strip(args.submit_folder.replace("/", ".") + "."), r.mu, r.sigma, submit_count[s]])
    submit_rating_vec.sort(key=lambda x: x[1], reverse=True)
    p_str = ""
    for index, s in enumerate(submit_rating_vec):
        p_str += f"{index + 1}: {s[0]} {s[1]:.3f} {s[2]:.3f}\n"
    if len(submit_rating_vec) > 0:
        table_txt = "# Scores\n\n"
        table_txt += f'Modified Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n\n'

        for r, sv in enumerate(submit_rating_vec):
            sv.insert(0, r + 1)
        table_txt += show_table(submit_rating_vec, list(range(len(submit_rating_vec[0]))),
                                ["rank", "submit", "score", "sigma", "pk_num"])
        if len(args.save_result) > 0:
            with open(args.save_result, "w") as f:
                f.write(table_txt)
    logging.info(f"rank cost: {int(1000 * (time.time() - start_time))}ms\n{p_str}\n")


def save_rank(submit_rating, json_files_set):
    with open("rank.tmp", "wb") as f:
        pickle.dump([submit_rating, json_files_set], f)


def main(args=get_args()):
    submits = find_submit(Path(args.submit_folder))
    logging.info(f"submits: {json.dumps(submits, indent=4)}")
    if len(submits) == 0:
        return

    all_datas_map = get_all_datas()

    submit_count = init_submit_count(submits, all_datas_map)

    submit_rating = {}
    json_files_set = set()
    rank_tmp_path = Path("rank.tmp")
    if rank_tmp_path.exists():
        submit_rating, json_files_set = pickle.load(rank_tmp_path.open("rb"))

    env = trueskill.TrueSkill(mu=1000)
    for submit in submits:
        if submit not in submit_rating:
            submit_rating[submit] = env.create_rating()
    for submit in set(submit_rating.keys()) - set(submits):
        del submit_rating[submit]

    if args.rank:
        rank(submit_rating, all_datas_map, env, json_files_set, submit_count)
        save_rank(submit_rating, json_files_set)

    if args.battle:
        for _ in range(args.battle_count):
            datas_map = collect_new_datas(all_datas_map, submit_count, args)
            if args.rank:
                rank(submit_rating, datas_map, env, json_files_set, submit_count)
                save_rank(submit_rating, json_files_set)


if __name__ == '__main__':
    main()
