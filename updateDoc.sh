# Updates the source documentation

cd docs
make html
cd ..
cp -r docs/_build/html/* docs/
