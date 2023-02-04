import styles from '../styles/Home.module.css'

export default function HomePage(this: any) {
    return (
        <div>
            <h1>Home Page</h1>
            <h2>File Upload</h2>
            <form id="uploadForm" onChange={this.uploadFile}>
                <input type="file" id="file" name="file"/>
                <input type="submit" value="Upload"/>
            </form>
            <footer className={styles.footer}>
                <p>Thanos Tsiamis @Utrecht University</p>
            </footer>

        </div>


    )
}
let uploadFile;
uploadFile = async (e: { target: { files: any[]; }; }) => {
    const file = e.target.files[0];
    if (file != null) {
        const data = new FormData();
        data.append('file_from_react', file);

        let response = await fetch('/upload_file',
            {
                method: 'post',
                body: data,
            }
        );
        let res = await response.json();
        if (res.status !== 1) {
            alert('Error uploading file');
        }
    }
};
