// Presenter tour: six stops. Each stop only sets the replay state; every sentence the tour
// shows about a bed is computed from the API response for that bed (see TourPage / TourRail).

export interface TourStop {
  n: number;
  title: string;
  bed?: string; // patient_id of the demo bed
  t: number;
  truth: boolean;
  route: "bed" | "ward" | "evidence";
  prompt: string; // what the presenter says or does here (no numbers; those come from the data)
}

export const TOUR: TourStop[] = [
  {
    n: 1,
    title: "A false alarm",
    bed: "p009767",
    t: 0,
    truth: true,
    route: "bed",
    prompt: "Start honest: this bed alerts in its first hour and never develops sepsis.",
  },
  {
    n: 2,
    title: "Caught early",
    bed: "p003071",
    t: 16,
    truth: true,
    route: "bed",
    prompt: "The first septic bed to cross the threshold. Read what the model saw, then what happened.",
  },
  {
    n: 3,
    title: "Caught very early",
    bed: "p119674",
    t: 30,
    truth: true,
    route: "bed",
    prompt: "An alert twelve or more hours before onset: useful, but very early alerts also cost attention.",
  },
  {
    n: 4,
    title: "Missed",
    bed: "p104996",
    t: 28,
    truth: true,
    route: "bed",
    prompt: "It does not catch everyone. This septic patient never crossed the threshold.",
  },
  {
    n: 5,
    title: "The threshold",
    t: 40,
    truth: false,
    route: "ward",
    prompt: "Move the alert threshold: lower catches more, earlier, with more false alarms; higher is quieter and misses more.",
  },
  {
    n: 6,
    title: "The evidence",
    t: 0,
    truth: false,
    route: "evidence",
    prompt: "The lockbox test, opened once for LightGBM, and the development comparison that chose the served ensemble.",
  },
];
