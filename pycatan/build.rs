use std::process::Command;
use std::fs;
use std::path::Path;


fn main() {

    let jsettlers_dir = "../external/jsettlers";

    let dest_dir = "python/pycatan";

    println!("cargo:rerun-if-changed={}/src", jsettlers_dir);

    println!("cargo:rerun-if-changed={}/build.gradle", jsettlers_dir);



    let status = Command::new("./gradlew")

        .arg("assemble")

        .current_dir(jsettlers_dir)

        .status()

        .expect("Failed to execute gradlew.");



    if !status.success() {

        panic!("Java build failed!");

    }

    let libs_dir = Path::new(jsettlers_dir).join("build/libs");
    let jars = ["JSettlers-2.7.00.jar", "JSettlersServer-2.7.00.jar"];

    fs::create_dir_all(dest_dir).expect("Failed to create destination directory");

    for jar in &jars {
        let src_path = libs_dir.join(jar);
        let dest_path = Path::new(dest_dir).join(jar);

        if src_path.exists() {
            fs::copy(&src_path, &dest_path)
                .expect(&format!("Failed to copy {}", jar));
        }
    }
}