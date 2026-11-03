class CampusProblem:
    def __init__(self, campus_graph, steps_config: dict, start_room: str = "gate",
                 profile=None, rule_pack_version: str = "v1", required_stations=None):
        """Reporting problem over the campus graph.

        With no profile (or explicit required_stations) every configured step is a
        goal, exactly as before. With a StudentProfile, the rules engine decides
        which stations that student must visit, and both goal_steps and
        steps_config are narrowed to them so search agents (which read either)
        keep working unchanged.
        """
        self.graph = campus_graph
        self.start_room = start_room
        self.profile = profile
        self.rule_pack_version = rule_pack_version
        self.all_steps_config = steps_config
        self.checklist = None

        if required_stations is None and profile is not None:
            from core.rules_engine import build_checklist
            self.checklist = build_checklist(profile, rule_pack_version)
            required_stations = self.checklist.required_stations

        if required_stations is None:
            self.steps_config = steps_config
        else:
            unknown = set(required_stations) - set(steps_config)
            if unknown:
                raise ValueError(
                    f"Required stations not in steps config: {sorted(unknown)}"
                )
            self.steps_config = {
                room: cfg for room, cfg in steps_config.items()
                if room in set(required_stations)
            }
        self.goal_steps = set(self.steps_config.keys())

    def initial_state(self):
        
        return (self.start_room, frozenset())

    def actions(self, state):
        #this will return valid actions for this state
        current_room, completed_steps = state
        valid_actions = []

        neighbors = self.graph.get_neighbors(current_room)
        for neighbor in neighbors:
            valid_actions.append(("move", neighbor))

        if current_room in self.steps_config and current_room not in completed_steps:
            valid_actions.append(("complete", current_room))

        return valid_actions

    def result(self, state, action):
        #resulting state from current chosen action
        current_room, completed_steps = state
        action_type, target = action

        if action_type == "move":
            return (target, completed_steps)
        elif action_type == "complete":
            new_completed = completed_steps | {target}
            return (current_room, new_completed)
        
        return state

    def goal_test(self, state):
        _, completed_steps = state
        return self.goal_steps.issubset(completed_steps)

    def path_cost(self, current_cost: float, state1, action, state2, travel_time: float = 0, wait_time: float = 0, service_time: float = 0) -> float:
        return current_cost + travel_time + wait_time + service_time