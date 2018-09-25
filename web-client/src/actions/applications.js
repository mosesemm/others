
export const STEP_TO="STEP_TO";

export const goToStep = (step) => {

    return {
        step,
        type: STEP_TO
    }
}