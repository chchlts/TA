import { TextField, Box, Button } from '@mui/material';
import { ChangeEvent, useEffect, useState } from 'react';

interface ImageInputProps {
    onFileChange?: (fileSrc: string) => void;
    onSubmit: (xRay: File) => void;
    loading?: boolean;
}

function ImageInput(props: ImageInputProps) {
    const [imageFile, setImageFile] = useState<FileList | null>(null);
    const submitDisabled = imageFile === null;

    const fileChangeHandler = (e: ChangeEvent<HTMLInputElement>) => {
        const files = e.currentTarget.files;
        setImageFile(files);

        if (props.onFileChange) {
            if (files) {
                const url = URL.createObjectURL(files[0]);
                props.onFileChange(url);
            }
        }
    }

    const submitHandler = () => {
        if (imageFile && imageFile!.length > 0) {
            props.onSubmit(imageFile[0]);
        }
    }


    return (
        <Box>
            <TextField
                type="file"
                name="brixia-image"
                size="small"
                fullWidth
                sx={{ mb: 1 }}
                onChange={fileChangeHandler}
            />
            <Button
                style={{
                    width: '135px',
                    height: '45px',
                }}
                variant="contained"
                disabled={submitDisabled || props.loading}
                onClick={submitHandler}
            >
                {!props.loading ? 'Submit' : 'Loading'}
            </Button>
        </Box>
    )
}

export default ImageInput