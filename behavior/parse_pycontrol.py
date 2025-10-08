import pandas as pd
import numpy as np


def make_readable(milliseconds, show_ms=True):
    seconds, milliseconds = divmod(int(milliseconds), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if show_ms:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class ParseBehavior:
    def __init__(self, prints, events):
        self.prints = prints
        self.events = events

        self.make_master_df()
        self.extract_nose_detections()

    def make_master_df(self):
        # dwell times
        left_dwell = self.extract_print_data("left_dwell")
        right_dwell = self.extract_print_data("right_dwell")
        center_dwell = self.extract_print_data("center_dwell")

        # blocks
        new_blocks = self.extract_print_data(
            "new_block", ["block", "trials_until_change", "next_block"], include_time=True
        )
        new_blocks = new_blocks.drop(["trials_until_change", "next_block"], axis=1)
        new_blocks["block_plain"] = new_blocks["block"]
        #new_blocks["block"] = new_blocks["block"].apply(make_block_html)
        self.blocks = new_blocks.copy()
        new_blocks = new_blocks.drop(["time"], axis=1)

        combined_df = self.extract_choices_and_streaks()
        for data in [left_dwell, right_dwell, center_dwell, new_blocks]:
            combined_df = combined_df.merge(data, on="trial", how="left")

        # add events
        for event_id in sorted({e.name for e in self.events}):
            event_df = self.extract_event(event_id)
            combined_df = combined_df.merge(event_df, on="time", how="outer")

        combined_df = combined_df.merge(self.extract_notes(), on="time", how="outer")

        syringe = self.extract_print_data(
            "syringe", ["left_vol", "right_vol"], drop_duplicates=False, include_time=True
        )
        syringe = syringe.drop("trial", axis=1)
        combined_df = combined_df.merge(syringe, on="time", how="outer")

        combined_df = combined_df.sort_values(by="time")
        combined_df = combined_df.reset_index(drop=True)

        # this will remove rows below the last completed trial record
        combined_df["trial"] = combined_df["trial"].bfill().ffill()

        combined_df[["left_vol", "right_vol"]] = combined_df[["left_vol", "right_vol"]].ffill()
        combined_df["left_delivered"] = combined_df["left_vol"].diff().shift(-1)
        combined_df["right_delivered"] = combined_df["right_vol"].diff().shift(-1)
        first_valid_index = combined_df["left_delivered"].first_valid_index()
        combined_df.loc[first_valid_index, "left_delivered"] = combined_df.loc[first_valid_index, "left_vol"]
        combined_df.loc[first_valid_index, "right_delivered"] = combined_df.loc[first_valid_index, "right_vol"]

        self.all_df = combined_df

        last_entry = self.all_df.iloc[len(self.all_df) - 1]
        self.completed_trials = int(last_entry.trial)
        self.max_vol = max(list(self.all_df.left_vol)[-1], list(self.all_df.right_vol)[-1]) / 1000
        self.max_dispense = max(combined_df["left_delivered"].max(), combined_df["right_delivered"].max())
        self.session_duration = make_readable(int(last_entry.time * 1000))

    def create_notion_multi_select(self, column_name):
        unique_array = self.all_df[column_name].dropna().unique()
        multi_select_list = []
        for val in sorted(unique_array):
            multi_select_list.append({"name": val.replace(",", "comma")})  # multi_select does not allow commas
        return multi_select_list

    def extract_print_data(self, key, data_column_names=None, include_time=False, drop_duplicates=True):
        """Extract data from data.prints. Print data is in the form T:{trial};{key};{data}
        Args:
            key: name of data being extracted
            data_column_names : column names that the data will be extracted to. Defaults to None.
            include_time : whether to include a time column in the returned dataframe

        Returns:
            dataframe: datframe that has a the trial and data
        """
        data_holder = []
        times = [tup.time for tup in self.prints if tup.string.find(f"{key};") > -1]
        trials = [int(tup.string.split(";")[0].split(":")[1]) for tup in self.prints if tup.string.find(f"{key};") > -1]
        if not data_column_names:
            data_column_names = [key]
        for index in range(len(data_column_names)):
            try:
                data_holder.append(
                    [eval(tup.string.split(";")[2])[index] for tup in self.prints if tup.string.find(f"{key};") > -1]
                )
            except TypeError:
                data_holder.append(
                    [eval(tup.string.split(";")[2]) for tup in self.prints if tup.string.find(f"{key};") > -1]
                )
        df = pd.DataFrame(list(zip(times, trials, *data_holder)), columns=["time", "trial"] + data_column_names)
        if drop_duplicates:
            df = df.drop_duplicates(subset="trial", keep="last")
        if not include_time:
            df = df.drop("time", axis=1)
        return df

    def extract_blocks(self, use_existing=True):
        """Extract blocks and parse block data into a tidy DataFrame.

        The repository stores block data in the prints under key 'new_block'.
        make_master_df already creates `self.blocks` (with a human-friendly
        HTML `block` column and an unmodified `block_plain` column). This
        method will read `self.blocks` (or re-extract the print data) and
        create `self.parsed_blocks`: a DataFrame with one row per sequence
        (sub-sequence) in each block including metadata (time, trial,
        trials_until_change, next_block).

        Returns:
            pandas.DataFrame: columns [time, trial, block_name, block_index,
            sequence_index, sequence_raw, sequence_str, trials_until_change,
            next_block]
        Also stores the dataframe on self.parsed_blocks for later use.
        """
        # obtain blocks DataFrame

        blocks_df = self.extract_print_data(
            "new_block", ["block", "trials_until_change", "next_block"], include_time=True
        )

        rows = []
        for _, r in blocks_df.iterrows():
            time = r.get("time")
            trial = r.get("trial")
            block_plain = r.get("block")
            trials_until_change = r.get("trials_until_change") if "trials_until_change" in r.index else np.nan

            # try to coerce string representations into Python objects only if needed
            if block_plain is None:
                bp = None
            elif isinstance(block_plain, (list, tuple)):
                bp = block_plain
            elif isinstance(block_plain, str):
                # try to eval a literal
                try:
                    bp = eval(block_plain)
                except Exception:
                    bp = block_plain
            else:
                # unknown type, keep as-is
                bp = block_plain

            # parse bp into sequences
            if isinstance(bp, (list, tuple)) and len(bp) >= 1:
                block_name = bp[0]
                sequences = bp[1:]
                for seq_idx, seq in enumerate(sequences, start=1):
                    seq_raw = seq
                    try:
                        seq_str = str(seq)
                    except Exception:
                        seq_str = repr(seq)
                    rows.append(
                        {
                            "time": time,
                            "trial": trial,
                            "block_name": block_name,
                            "sequence_index": seq_idx,
                            "sequence_raw": seq_raw[0],
                            "reward_vol": seq_raw[1],
                            "p_reward": seq_raw[2],
                            "repeat": seq_raw[3],
                            "stay_after_reward": seq_raw[4],
                            "trials_until_change": trials_until_change,
                            "sequence_str": seq_str,
                        }
                    )

        parsed = pd.DataFrame(rows)
        # store for later use
        self.parsed_blocks = parsed

        return parsed

    def extract_notes(self):
        note_list = [
            (tup.time, eval(tup.string)[0].replace("|", "\n"), eval(tup.string)[1])
            for tup in self.prints
            if tup.subtype == "user"
        ]

        self.all_notes_str = ""
        for _, note, author in note_list:
            self.all_notes_str += f"{author}: {note}\n"

        return pd.DataFrame(note_list, columns=["time", "note", "author"])
    
    def extract_event(self, event_name):
        event_list = [(tup.time, True) for tup in self.events if tup.name == event_name]
        return pd.DataFrame(event_list, columns=["time", event_name])

    def extract_choices_and_streaks(self):
        result_df = self.extract_print_data(
            "result",
            [
                "choice",
                "outcome",
                "outcome_delay",
                "reward_volume",
                "matched_str",
                "matched_rule",
            ],
            include_time=True,
        )
        result_df.reward_volume = result_df.reward_volume.fillna(0)
        result_df["matched_str_length"] = result_df.matched_str.str.len() - 1
        # add streaks
        choice_ltrs = result_df.choice
        leftMask = choice_ltrs == "L"
        rightMask = choice_ltrs == "R"
        streak_DF = choice_ltrs.copy().to_frame()
        streak_DF["streak_start"] = streak_DF.ne(streak_DF.shift())
        streak_DF["streak_id"] = streak_DF["streak_start"].cumsum()
        streak_DF["streak_counts"] = streak_DF.groupby("streak_id").cumcount() + 1
        streak_DF.loc[leftMask, "streak_counts"] = streak_DF.loc[leftMask, "streak_counts"] * 1
        streak_DF.loc[rightMask, "streak_counts"] = streak_DF.loc[rightMask, "streak_counts"] * -1
        streak_DF.drop(columns=["choice", "streak_start", "streak_id"], inplace=True)
        return pd.concat([result_df, streak_DF], axis=1)
    

    def extract_nose_detections(self):
        for nose in ["l_nose", "c_nose", "r_nose"]:
            in_mask = self.all_df[f"{nose}_TOF_in"] == True
            out_mask = self.all_df[f"{nose}_TOF_out"] == True

            first_in = in_mask.idxmax()
            first_out = out_mask.idxmax()

            last_in = in_mask[::-1].idxmax()
            last_out = out_mask[::-1].idxmax()

            if first_in > first_out:  # nose was already in the poke when session began
                in_mask.loc[first_in] = False
            if last_in > last_out:  # nose was still in the poke when the session ended
                in_mask.loc[last_in] = False

            time_out = np.array(self.all_df[out_mask].time)
            time_in = np.array(self.all_df[in_mask].time)
            duration = time_out - time_in

            # self.all_df[f"{nose}_durations"] = np.nan
            self.all_df.loc[in_mask, f"{nose}_time_in"] = time_in
            self.all_df.loc[in_mask, f"{nose}_time_out"] = time_out
            self.all_df.loc[in_mask, "durations"] = duration
    
    def extract_nose_in_out_times(self):
        """For each trial, find the time of the last center nose TOF out and the
        time of the last side nose TOF out (left or right) depending on the
        trial's choice ('L' or 'R').

        Returns:
            pandas.DataFrame: columns ['trial','choice','last_c_nose_out',
            'last_side_nose_out','side'] with one row per trial. Also stores
            the dataframe on self.last_nose_outs and merges the three new
            columns into self.all_df (left-joined on 'trial').
        """
        rows = []
        # gather unique completed trials in sorted order
        trials = [t for t in sorted(self.all_df.trial.dropna().unique())]

        for t in trials:
            trial_num = int(t)
            trial_rows = self.all_df[self.all_df.trial == trial_num]

            # determine the trial's recorded choice (if any)
            choice_vals = trial_rows.choice.dropna().unique()
            choice = choice_vals[0] if len(choice_vals) > 0 else None
            side = choice.lower() if choice else None

            # first center nose in within this trial
            # initialize defaults
            first_c = np.nan
            first_side = np.nan

            # first center nose in within this trial
            if "c_nose_TOF_in" in trial_rows.columns:
                center_in_times = trial_rows.loc[trial_rows["c_nose_TOF_in"] == True, "time"]
                first_c = float(center_in_times.min()) if not center_in_times.empty else np.nan
            
            # last center nose out within this trial
            if "c_nose_TOF_out" in trial_rows.columns:
                center_times = trial_rows.loc[trial_rows["c_nose_TOF_out"] == True, "time"]
                last_c = float(center_times.max()) if not center_times.empty else np.nan
            else:
                last_c = np.nan

            # first side nose in within this trial (based on choice)
            if side:
                col = f"{side}_nose_TOF_in"
                if col in trial_rows.columns:
                    side_in_times = trial_rows.loc[trial_rows[col] == True, "time"]
                    # choose the earliest side-in strictly after first_c if first_c exists
                    if not pd.isna(first_c) and not side_in_times.empty:
                        later = side_in_times[side_in_times > first_c]
                        first_side = float(later.min()) if not later.empty else np.nan
                    else:
                        # fallback: if first_c is nan, take the earliest side-in if any
                        first_side = float(side_in_times.min()) if not side_in_times.empty else np.nan
                else:
                    first_side = np.nan
            else:
                first_side = np.nan

            # last side nose out within this trial (based on choice)
            trial_rows = self.all_df[self.all_df.trial == trial_num + 1] # look in next trial for side nose out
            if side:
                col = f"{side}_nose_TOF_out"
                if col in trial_rows.columns:
                    side_times = trial_rows.loc[trial_rows[col] == True, "time"]
                    last_side = float(side_times.max()) if not side_times.empty else np.nan
                else:
                    last_side = np.nan
            else:
                last_side = np.nan

            rows.append(
                {
                    "trial": trial_num,
                    "choice": choice,
                    "c_in": first_c,
                    "side_in": first_side,
                    "c_out": last_c,
                    "side_out": last_side,
                    "side": side,
                }
            )

            # verification: check ordering c_in < c_out < side_in < side_out
            issue = ""
            order_ok = True
            vals = {"c_in": first_c, "c_out": last_c, "side_in": first_side, "side_out": last_side}
            # check for missing values
            missing = [k for k, v in vals.items() if pd.isna(v)]
            if missing:
                order_ok = False
                issue = f"missing:{','.join(missing)}"
            else:
                if not (first_c < last_c):
                    order_ok = False
                    issue = "c_in>=c_out"
                elif not (last_c < first_side):
                    order_ok = False
                    issue = "c_out>=side_in"
                elif not (first_side < last_side):
                    order_ok = False
                    issue = "side_in>=side_out"

            # additional verification: current first_c should be greater than
            # the previous trial's side_out (if previous exists)
            prev_last_side = np.nan
            if len(rows) >= 2:
                prev_last_side = rows[-2].get("side_out", np.nan)

            if not pd.isna(prev_last_side):
                if pd.isna(first_c) or not (first_c > prev_last_side):
                    # mark as violation; append to issue if one already exists
                    order_ok = False
                    add_issue = "first_c<=prev_side_out"
                    if issue:
                        issue = f"{issue};{add_issue}"
                    else:
                        issue = add_issue

            # attach verification info to last appended row (mutable dict in rows)
            rows[-1]["order_ok"] = order_ok
            rows[-1]["order_issue"] = issue

        df = pd.DataFrame(rows)
        # store for later use
        self.nose_in_out_times = df
        return df

    def extract_reward_flags(self):
        """Create a dataframe `reward` with trial numbers and boolean columns
        derived from the `outcome` column in the choices dataframe.

        Columns created:
            reward      -> outcome == 'C'
            withheld    -> outcome == 'W'
            no_reward   -> outcome == 'N'
            predicted   -> outcome == 'P'
            background  -> outcome == 'B'
            abandoned   -> outcome == 'A'

        The function uses `extract_choices_and_streaks()` to obtain outcomes
        with their trial numbers and returns the resulting dataframe. The
        dataframe is also stored as `self.reward`.
        """
        choices = self.extract_choices_and_streaks()
        # ensure we have trial and outcome
        if "trial" not in choices.columns or "outcome" not in choices.columns:
            raise RuntimeError("choices dataframe missing 'trial' or 'outcome' columns")

        reward_df = pd.DataFrame()
        reward_df["trial"] = choices["trial"].astype(int)
        outcome = choices["outcome"].fillna("")

        reward_df["reward"] = outcome == "C"
        reward_df["withheld"] = outcome == "W"
        reward_df["no_reward"] = outcome == "N"
        reward_df["predicted"] = outcome == "P"
        reward_df["background"] = outcome == "B"
        reward_df["abandoned"] = outcome == "A"

        # store for later use
        self.reward = reward_df
        return reward_df

    