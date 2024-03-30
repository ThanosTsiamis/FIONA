import React, {useCallback, useState} from "react";
import {styled, Theme} from '@mui/material/styles';
import Switch from "@mui/material/Switch";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

const FancySwitch = styled(Switch)(({theme}: { theme: Theme }) => ({
    width: 42,
    height: 26,
    padding: 0,
    '& .MuiSwitch-switchBase': {
        padding: 0,
        margin: 2,
        transitionDuration: '300ms',
        '& .MuiSwitch-thumb': {
            width: 22,
            height: 22,
            backgroundColor: theme.palette.mode === 'dark' ? '#003892' : '#001e3c',
            boxShadow: 'none',
        },
        '&.Mui-checked': {
            transform: 'translateX(16px)',
            color: '#ffffff',
            '& + .MuiSwitch-track': {
                backgroundColor: theme.palette.mode === 'dark' ? '#52d869' : '#52d869',
                opacity: 1,
                border: 'none',
            },
        },
        '&.Mui-focusVisible .MuiSwitch-thumb': {
            color: '#52d869',
            border: '6px solid #fff',
        },
        '&.Mui-disabled .MuiSwitch-thumb': {
            color: theme.palette.mode === 'dark' ? '#003892' : '#001e3c',
        },
        '&.Mui-disabled + .MuiSwitch-track': {
            opacity: theme.palette.mode === 'dark' ? 0.2 : 0.5,
        },
    },
    '& .MuiSwitch-track': {
        borderRadius: 26 / 2,
        backgroundColor: theme.palette.mode === 'dark' ? '#8796A5' : '#aab4be',
        opacity: 1,
        transition: theme.transitions.create(['background-color'], {
            duration: 500,
        }),
    },
}));

export const ToggleSwitch = () => {
    const [collapsed, setCollapsed] = useState(false);

    const toggleCollapsed = useCallback(() => {
        setCollapsed((previouslyCollapsed) => {
            return !previouslyCollapsed;
        });
    }, []);

    return (
        <Stack direction="row" component="label" alignItems="center" justifyContent="center">
            <Typography>
                Brief
            </Typography>
            <FancySwitch onChange={toggleCollapsed} checked={collapsed}/>
            <Typography>
                Detailed
            </Typography>
        </Stack>
    );
};

export default ToggleSwitch;