import styles from '../styles/Home.module.css'
import axios from "axios";

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

