import { Box, Stack, Typography } from '@mui/material';

interface ScoreResultProps {
    colorScore : number;
    colorDesc:string;
    exudateScore : number;
    exudateDesc : string;
    inflammationScore : number;
    inflammationDesc : string;
    severity: string;
    show?: boolean;
}

function ScoreResult(props: ScoreResultProps) {

    if (!props.show) {
        return null;
    }

    return (
        <Box>
            <Typography mt={2}>
                Status luka : <strong>{props.severity}</strong>
            </Typography>
            <Stack direction="row">
                <table
                    border={1}
                    style={{ borderCollapse: 'collapse', alignSelf: 'flex-start', marginRight: '24px' }}
                >
                    <tbody>
                        <tr>
                            <td style={{ padding: '1rem'}}><strong>Indikator</strong></td>
                            <td style={{ padding: '1rem'}}><strong>Skor</strong></td>
                            <td style={{ padding: '1rem'}}><strong>Deskripsi</strong></td>
                        </tr>
                        <tr>
                            <td style={{ padding: '1rem' }}>Warna</td>
                            <td style={{ padding: '1rem' }}><strong> {props.colorScore} </strong></td>
                            <td style={{ padding: '1rem' }}>{props.colorDesc}</td>
                        </tr>
                        <tr>
                            <td style={{ padding: '1rem' }}>Eksudat</td>
                            <td style={{ padding: '1rem' }}> <strong>{props.exudateScore}</strong></td>
                            <td style={{ padding: '1rem' }}>{ props.exudateDesc }</td>
                        </tr>
                        <tr>
                            <td style={{ padding: '1rem' }}>Inflamasi</td>
                            <td style={{ padding: '1rem' }}><strong>{props.inflammationScore}</strong></td>                            
                            <td style={{ padding: '1rem' }}>{props.inflammationDesc}</td>
                        </tr>
                    </tbody>
                </table>
                
            </Stack>
            
            <Typography mt={2}>
                <span style={{ color: "red" }}>*</span> Status merupakan hasil dari penjumlahan skor untuk seluruh indikator.
            </Typography>
            <Typography><strong>16 - 10 </strong> : Berat</Typography>
            <Typography><strong>9 - 8 </strong> : Sedang</Typography>
            <Typography><strong>7 - 5 </strong> : Ringan</Typography>
            <Typography mb={2}><strong>3 - 4 </strong> : Sembuh</Typography>
        </Box>
    )
}

export default ScoreResult