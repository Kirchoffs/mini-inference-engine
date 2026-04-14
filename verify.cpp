#include <iostream>
#include <armadillo>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sentencepiece_processor.h>

int main(int argc, char** argv) {
    // Initialize and verify Google Logging
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    LOG(INFO) << "Checking libraries...";

    // Verify Armadillo
    arma::mat matrix = arma::randu<arma::mat>(3, 3);
    std::cout << "Armadillo Matrix:\n" << matrix << std::endl;

    // Verify GoogleTest
    std::cout << "GTest Version: " << GTEST_VERSION << std::endl;

    // Verify SentencePiece
    sentencepiece::SentencePieceProcessor sp;
    std::cout << "SentencePiece Processor initialized successfully." << std::endl;
    LOG(INFO) << "SentencePiece version: " << SP_VERSION;

    LOG(INFO) << "Environment verification SUCCESS!";
    
    return 0;
}
