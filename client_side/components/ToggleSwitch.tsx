import React, {useState, useCallback} from "react";
import Switch from "@mui/material/Switch";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

export const ToggleSwitch = () => {
    const [collapsed, setCollapsed] = useState(false);

    const toggleCollapsed = useCallback(() => {
        setCollapsed(previouslyCollapsed => {
            return !previouslyCollapsed;
        });
    }, []);

    return (
        <Stack direction="row" component="label" alignItems="center" justifyContent="center">
            <Typography>
                Brief
            </Typography>
            <Switch onChange={toggleCollapsed} value={collapsed}/>
            <Typography>
                Detailed
            </Typography>
        </Stack>
    );
}
export default ToggleSwitch;
