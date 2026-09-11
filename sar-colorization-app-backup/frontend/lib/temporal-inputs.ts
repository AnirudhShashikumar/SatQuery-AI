export interface TemporalDates {
  primary: string;
  secondary: string;
}

export function swapTemporalDates({ primary, secondary }: TemporalDates): TemporalDates {
  return { primary: secondary, secondary: primary };
}
