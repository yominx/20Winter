int main(int argc, char * argv[]) {

    namedWindow(window_name, WINDOW_AUTOSIZE);
    cv::Mat image = imread("./SIFT/myimage.jpg", IMREAD_COLOR);

    cout << "Size of the picture is" << image.size() << endl;
    keyPoints.clear();
    makeOctave(image, image.size());
    DetectExtrema();
    AssignOrientations();
    ExtractKeypointDescriptors();
    cout << "NO ERROR" << endl;
    showFeatures(&image);
    waitKey(0);
    
    return 0;
    }

int main(int argc, char * argv[]) try{
    rs2::colorizer color_map;
    rs2::pipeline pipe;
    pipe.start();

    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0 && frame){

        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        rs2::frame  rgbImg = data.get_color_frame();
        Mat image(rgbImg.size(), CV_8UC3, (void*)rgbImg.get_data(), Mat::AUTO_STEP);
        keyPoints.clear();

        makeOctave(image, image.size());
        DetectExtrema();
        AssignOrientations();
        ExtractKeypointDescriptors();
        showFeatures(&image);
        }

    return EXIT_SUCCESS;
    }
catch (const rs2::error & e){
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
    }
catch (const std::exception& e){
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
    }