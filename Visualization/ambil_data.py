import csv
import os
import pickle
from itertools import zip_longest


def pkl_multi_agent_to_csv(input_pkl_path, output_csv_path=None):
	"""Convert multi-agent PKL data ('data_agent') to a flat CSV table."""
	if output_csv_path is None:
		base_name = os.path.splitext(input_pkl_path)[0]
		output_csv_path = f"{base_name}.csv"

	with open(input_pkl_path, "rb") as f:
		loaded = pickle.load(f)

	if "data_agent" not in loaded:
		raise KeyError("Key 'data_agent' tidak ditemukan di file PKL.")

	data_agent = loaded["data_agent"]

	headers = [
		"agent_id",
		"t",
		"x",
		"y",
		"z",
		"vx",
		"vy",
		"vz",
		"mode",
		"scale",
	]

	with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(headers)

		for idx, agent in enumerate(data_agent, start=1):
			path = agent.get("path")
			if path is None:
				continue

			for row in path:
				writer.writerow([idx, *row.tolist()])

	return output_csv_path


AGENT_HEADERS = [
	"t",
	"x",
	"y",
	"z",
	"vx",
	"vy",
	"vz",
	"mode",
	"scale",
]


QUAD_HEADERS = [
	"t",
	"x",
	"y",
	"z",
	"q0",
	"q1",
	"q2",
	"q3",
	"phi",
	"theta",
	"psi",
	"v_x",
	"v_y",
	"v_z",
	"p",
	"q",
	"r",
	"wM1",
	"wM2",
	"wM3",
	"wM4",
	"x_sp",
	"y_sp",
	"z_sp",
	"q0_des",
	"q1_des",
	"q2_des",
	"q3_des",
	"phi_sp",
	"theta_sp",
	"psi_sp",
	"v_x_sp",
	"v_y_sp",
	"v_z_sp",
	"p_des",
	"q_des",
	"r_des",
	"yaw_rate_sp",
	"x_thr_sp",
	"y_thr_sp",
	"z_thr_sp",
	"wM1_sp",
	"wM2_sp",
	"wM3_sp",
	"wM4_sp",
]


def _normalize_time_key(t_value, decimals=8):
	"""Normalize float time so matching between files is more stable."""
	return round(float(t_value), decimals)


def _flatten_agent_by_time(agent_loaded, decimals=8):
	"""Return {time: [rows]} where each row is {'id': int, 'agent': dict}."""
	if "data_agent" not in agent_loaded:
		raise KeyError("Key 'data_agent' tidak ditemukan di file PKL agent.")

	by_time = {}
	for idx, agent in enumerate(agent_loaded["data_agent"], start=1):
		path = agent.get("path")
		if path is None:
			continue

		for row in path:
			values = row.tolist()
			t_key = _normalize_time_key(values[0], decimals=decimals)
			by_time.setdefault(t_key, []).append(
				{
					"id": idx,
					"agent": dict(zip(AGENT_HEADERS, values)),
				}
			)

	for t_key in by_time:
		by_time[t_key].sort(key=lambda item: item["id"])

	return by_time


def _flatten_quad_by_time(quad_loaded, decimals=8):
	"""Return {time: [rows]} where each row is {'id': int, 'quad': dict}."""
	if "data_quad" not in quad_loaded:
		raise KeyError("Key 'data_quad' tidak ditemukan di file PKL quad.")

	by_time = {}
	for idx, quad in enumerate(quad_loaded["data_quad"], start=1):
		path = quad.get("path")
		quad_id = quad.get("id", idx)
		if path is None:
			continue

		for row in path:
			values = row.tolist()
			t_key = _normalize_time_key(values[0], decimals=decimals)
			by_time.setdefault(t_key, []).append(
				{
					"id": quad_id,
					"quad": dict(zip(QUAD_HEADERS, values)),
				}
			)

	for t_key in by_time:
		by_time[t_key].sort(key=lambda item: item["id"])

	return by_time


def pkl_merge_agent_quad_by_time_to_csv(agent_pkl_path, quad_pkl_path, output_csv_path=None):
	"""
	Merge data_agent + data_quad into one CSV with primary grouping by time.

	At each timestamp, rows are aligned by order of id (ascending) and merged.
	The 'id' is written as a regular column (not used as the merge key).
	"""
	if output_csv_path is None:
		agent_base = os.path.splitext(os.path.basename(agent_pkl_path))[0]
		quad_base = os.path.splitext(os.path.basename(quad_pkl_path))[0]
		output_csv_path = f"merged_{agent_base}__{quad_base}.csv"

	with open(agent_pkl_path, "rb") as f:
		agent_loaded = pickle.load(f)

	with open(quad_pkl_path, "rb") as f:
		quad_loaded = pickle.load(f)

	agent_by_time = _flatten_agent_by_time(agent_loaded)
	quad_by_time = _flatten_quad_by_time(quad_loaded)

	all_times = sorted(set(agent_by_time.keys()) | set(quad_by_time.keys()))

	headers = [
		"id",
		"t",
		"agent_x",
		"agent_y",
		"agent_z",
		"agent_vx",
		"agent_vy",
		"agent_vz",
		"agent_mode",
		"agent_scale",
		"quad_x",
		"quad_y",
		"quad_z",
		"quad_q0",
		"quad_q1",
		"quad_q2",
		"quad_q3",
		"quad_phi",
		"quad_theta",
		"quad_psi",
		"quad_v_x",
		"quad_v_y",
		"quad_v_z",
		"quad_p",
		"quad_q",
		"quad_r",
		"quad_wM1",
		"quad_wM2",
		"quad_wM3",
		"quad_wM4",
		"quad_x_sp",
		"quad_y_sp",
		"quad_z_sp",
		"quad_q0_des",
		"quad_q1_des",
		"quad_q2_des",
		"quad_q3_des",
		"quad_phi_sp",
		"quad_theta_sp",
		"quad_psi_sp",
		"quad_v_x_sp",
		"quad_v_y_sp",
		"quad_v_z_sp",
		"quad_p_des",
		"quad_q_des",
		"quad_r_des",
		"quad_yaw_rate_sp",
		"quad_x_thr_sp",
		"quad_y_thr_sp",
		"quad_z_thr_sp",
		"quad_wM1_sp",
		"quad_wM2_sp",
		"quad_wM3_sp",
		"quad_wM4_sp",
	]

	with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(headers)

		for t_key in all_times:
			agent_rows = agent_by_time.get(t_key, [])
			quad_rows = quad_by_time.get(t_key, [])

			for agent_item, quad_item in zip_longest(agent_rows, quad_rows, fillvalue=None):
				agent_dict = agent_item["agent"] if agent_item else {}
				quad_dict = quad_item["quad"] if quad_item else {}

				# Keep id as a plain output field; not a required merge key.
				row_id = None
				if agent_item is not None:
					row_id = agent_item["id"]
				elif quad_item is not None:
					row_id = quad_item["id"]

				writer.writerow(
					[
						row_id,
						t_key,
						agent_dict.get("x"),
						agent_dict.get("y"),
						agent_dict.get("z"),
						agent_dict.get("vx"),
						agent_dict.get("vy"),
						agent_dict.get("vz"),
						agent_dict.get("mode"),
						agent_dict.get("scale"),
						quad_dict.get("x"),
						quad_dict.get("y"),
						quad_dict.get("z"),
						quad_dict.get("q0"),
						quad_dict.get("q1"),
						quad_dict.get("q2"),
						quad_dict.get("q3"),
						quad_dict.get("phi"),
						quad_dict.get("theta"),
						quad_dict.get("psi"),
						quad_dict.get("v_x"),
						quad_dict.get("v_y"),
						quad_dict.get("v_z"),
						quad_dict.get("p"),
						quad_dict.get("q"),
						quad_dict.get("r"),
						quad_dict.get("wM1"),
						quad_dict.get("wM2"),
						quad_dict.get("wM3"),
						quad_dict.get("wM4"),
						quad_dict.get("x_sp"),
						quad_dict.get("y_sp"),
						quad_dict.get("z_sp"),
						quad_dict.get("q0_des"),
						quad_dict.get("q1_des"),
						quad_dict.get("q2_des"),
						quad_dict.get("q3_des"),
						quad_dict.get("phi_sp"),
						quad_dict.get("theta_sp"),
						quad_dict.get("psi_sp"),
						quad_dict.get("v_x_sp"),
						quad_dict.get("v_y_sp"),
						quad_dict.get("v_z_sp"),
						quad_dict.get("p_des"),
						quad_dict.get("q_des"),
						quad_dict.get("r_des"),
						quad_dict.get("yaw_rate_sp"),
						quad_dict.get("x_thr_sp"),
						quad_dict.get("y_thr_sp"),
						quad_dict.get("z_thr_sp"),
						quad_dict.get("wM1_sp"),
						quad_dict.get("wM2_sp"),
						quad_dict.get("wM3_sp"),
						quad_dict.get("wM4_sp"),
					]
				)

	return output_csv_path


if __name__ == "__main__":
	input_file = "multi_agent_data_scheme2_GUST_erc_formation1_NED.pkl"
	output_file = pkl_multi_agent_to_csv(input_file)
	print(f"Berhasil konversi: {input_file} -> {output_file}")

	# Contoh gabung dua PKL berdasarkan time (t)
	agent_file = "multi_agent_data_scheme2_GUST_erc_formation1_NED.pkl"
	quad_file = "multi_quad_data_scheme2_GUST_erc_formation1_NED.pkl"
	merged_file = pkl_merge_agent_quad_by_time_to_csv(agent_file, quad_file)
	print(f"Berhasil gabung: {agent_file} + {quad_file} -> {merged_file}")
