import React from 'react'
import { Box, Stack, Typography } from '@mui/material';

interface WoundImageProps {
    imgUrl: string;
}

function WoundImage(props: WoundImageProps) {

    if (!props.imgUrl) {
        return (
            <Box sx={{ width: '512px', height: '512px' }}>
                <Stack sx={{ width: '480px', height: '480px', bgcolor: "rgba(25, 118, 210, .1)", margin: '16px auto 0' }} justifyContent="center" alignItems="center">
                    <Typography variant="body1">Please Input Image</Typography>
                </Stack>

            </Box>
        )
    }

    return (
        <Box sx={{ width: '512px', height: '512px' }}>
            <img
                src={props.imgUrl}
                alt=""
                style={{ display: 'block', width: '480px', height: '480px', margin: '16px auto 0', objectFit: 'cover' }}
            />
        </Box>
    )
}

export default WoundImage