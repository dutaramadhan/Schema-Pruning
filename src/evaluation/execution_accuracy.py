from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import sys
from pathlib import Path

from func_timeout import func_timeout, FunctionTimedOut


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_ROOT = PROJECT_ROOT / "Dataset" / "BIRD" / "dev" / "dev_databases" / "dev_databases"
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "Dataset" / "BIRD" / "dev" / "dev.sql"
DEFAULT_DIFF_JSON = PROJECT_ROOT / "Dataset" / "BIRD" / "dev" / "dev.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "sql_eval" / "execution_accuracy.txt"

exec_result = []


def load_json(dir):
    with open(dir, "r", encoding="utf-8") as j:
        contents = json.loads(j.read())
    return contents


def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_place),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [("timeout",)]
        res = 0
    except Exception:
        result = [("error",)]
        res = 0
    result = {"sql_idx": idx, "res": res}
    return result


def package_sqls(sql_path, db_root_path, mode="gpt", data_mode="dev"):
    clean_sqls = []
    db_path_list = []
    if mode == "gpt":
        sql_data = json.load(open(sql_path, "r", encoding="utf-8"))
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split("\t----- bird -----\t")
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")
    elif mode == "gt":
        sqls = open(sql_path, encoding="utf-8")
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split("\t")
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")

    return clean_sqls, db_path_list


def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out),
            callback=result_callback,
        )
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])


def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    results = [res["res"] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i, content in enumerate(contents):
        if content["difficulty"] == "simple":
            simple_results.append(exec_results[i])

        if content["difficulty"] == "moderate":
            moderate_results.append(exec_results[i])

        if content["difficulty"] == "challenging":
            challenging_results.append(exec_results[i])

    simple_acc = sum([res["res"] for res in simple_results]) / len(simple_results)
    moderate_acc = sum([res["res"] for res in moderate_results]) / len(moderate_results)
    challenging_acc = sum([res["res"] for res in challenging_results]) / len(challenging_results)
    all_acc = sum(results) / num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists


def print_data(score_lists, count_lists):
    levels = ["simple", "moderate", "challenging", "total"]
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *count_lists))

    print("======================================    ACCURACY    =====================================")
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format("accuracy", *score_lists))


def save_results_to_file(score_lists, count_lists, filepath):
    levels = ["simple", "moderate", "challenging", "total"]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("{:20} {:20} {:20} {:20} {:20}\n".format("", *levels))
        f.write("{:20} {:<20} {:<20} {:<20} {:<20}\n".format("count", *count_lists))
        f.write("======================================    ACCURACY    =====================================\n")
        f.write("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}\n".format("accuracy", *score_lists))
        f.write("===========================================================================================\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIRD execution accuracy evaluator")
    parser.add_argument(
        "--predicted-sql-path",
        type=str,
        required=True,
        help="Path file prediksi JSON format BIRD: sql + '\\t----- bird -----\\t' + db_id",
    )
    parser.add_argument("--db-root-path", type=str, default=str(DEFAULT_DB_ROOT))
    parser.add_argument("--ground-truth-path", type=str, default=str(DEFAULT_GROUND_TRUTH))
    parser.add_argument("--diff-json-path", type=str, default=str(DEFAULT_DIFF_JSON))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--mode-predict", type=str, default="gpt", choices=["gpt"])
    parser.add_argument("--data-mode", type=str, default="dev")
    parser.add_argument("--meta-time-out", type=float, default=30.0)
    parser.add_argument("--num-cpus", type=int, default=16)
    return parser.parse_args()


def _ensure_db_root_suffix(db_root_path: str) -> str:
    if db_root_path.endswith("/") or db_root_path.endswith("\\"):
        return db_root_path
    return db_root_path + os.sep


if __name__ == "__main__":
    args = _parse_args()
    exec_result = []

    predicted_sql_path = args.predicted_sql_path
    db_root_path = _ensure_db_root_suffix(args.db_root_path)
    mode_predict = args.mode_predict
    data_mode = args.data_mode
    ground_truth_path = args.ground_truth_path
    meta_time_out = args.meta_time_out
    num_cpus = args.num_cpus
    diff_json_path = args.diff_json_path
    output_path = args.output_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pred_queries, db_paths = package_sqls(
        predicted_sql_path,
        db_root_path,
        mode=mode_predict,
        data_mode=data_mode,
    )

    gt_queries, db_paths_gt = package_sqls(
        ground_truth_path,
        db_root_path,
        mode="gt",
        data_mode=data_mode,
    )

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(
        query_pairs,
        db_places=db_paths,
        num_cpus=num_cpus,
        meta_time_out=meta_time_out,
    )
    exec_result = sort_results(exec_result)

    print("start calculate")
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(
        exec_result,
        diff_json_path,
    )
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    save_results_to_file(score_lists, count_lists, output_path)
    print("===========================================================================================")
    print("Finished evaluation")
