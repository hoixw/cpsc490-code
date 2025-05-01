# MMLU-Pro (Augmented)

This folder contains the code to build the augmented dataset. As it isn't strictly reproducible, the final dataset is included as well. Still, the code to reconstruct a similar dataset is included. All data files are contained in the `data/` subfolder.

## Splits

To explain how to construct the dataset, first, consider the original dataset:

| Discipline | Number of Questions | From Original MMLU | Newly Added |
|------------|---------------------|-------------------|-------------|
| Math | 1351 | 846 | 505 |
| Physics | 1299 | 411 | 888 |
| Chemistry | 1132 | 178 | 954 |
| Law | 1101 | 1101 | 0 |
| Engineering | 969 | 67 | 902 |
| Other | 924 | 924 | 0 |
| Economics | 844 | 444 | 400 |
| Health | 818 | 818 | 0 |
| Psychology | 798 | 493 | 305 |
| Business | 789 | 155 | 634 |
| Biology | 717 | 219 | 498 |
| Philosophy | 499 | 499 | 0 |
| Computer Science |410 |274 | 136 |
| History | 381 | 381 | 0 |

Excluding 'Other', there are 13 disciplines, and we will consider the 10 most frequent. Thus, our discipline with the least questions is 'Biology', with 717 queries. We utilise 200 'test' queries, and place the rest of the prompts into `train.csv`, except for math and physics, where we place 1000 of the prompts.

. This can be done with `conv-and-split.py`, which takes the original `parquet` file from [HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) and converts it into `train-orig.csv` and `test.csv`.

## Augmenting

Now, our dataset is significantly imbalanced — math and physics have 1000 training prompts, and other disciplines have significantly less (to varying degrees). To fix this, we augment questions for each discipline as needed. This is done by querying `GPT-4o` using a structured output format, asking it to augment the question and answer choices, whilst keeping the specifics of the question the same, and the order of the answers the same. We then shuffle the answer choices in the augmented question.

We augment each question at most once.

Here is the breakdown of the resulting dataset:

| Discipline | Original Training Questions | Augmented | Total |
|------------|---------------------|-------------------|-------------|
| Math | 1000 | 0 | 1000 |
| Physics | 1000 | 0 | 1000 |
| Chemistry | 932 | 68 | 1000 |
| Law | 901 | 99 | 1000 |
| Engineering | 769 | 231 | 1000 |
| Economics | 644 | 356 | 1000 |
| Health | 618 | 382 | 1000 |
| Psychology | 598 | 402 | 1000 |
| Business | 589 | 411 | 1000 |
| Biology | 517 | 483 | 1000 |


### Example of Original vs Augmented Question

#### Original Question
> **Question:** A community bank consists of four branch offices with approximately 60 employees each. The general management would like to institute 2 health risk-reduction program by encouraging weight loss and smoking cessation among the employees. Which of the following programs would be most effective?
>
> **Options:**
>
> 1. Development of, and participation in, local community group sessions focusing on weight loss and smoking cessation
> 2. **Employee reimbursement for costs and fees associated with professional help for weight loss and smoking cessation** ✓
> 3. A competition among the four branches focusing on stepwise reductions in weight and smoking
> 4. Providing gym memberships and nicotine patches to all employees
> 5. Instituting a penalty system for employees who do not lose weight or quit smoking
> 6. Mandatory weight loss and smoking cessation classes during work hours
> 7. Distribution of health information and self-help materials related to weight loss and smoking cessation
> 8. Implementing a company-wide ban on smoking and unhealthy food in the office.

#### Augmented Version
> **Question:** A local bank has four branches, each with about 60 staff members. Management wants to implement two health risk-reduction initiatives by promoting weight loss and quitting smoking among employees. Which of the following programs would likely be the most effective?
>
> **Options:**
> 1. Offering gym memberships and nicotine patches to every staff member
> 2. Enacting a company-wide prohibition on smoking and unhealthy foods in the workplace
> 3. Providing educational materials and self-help resources on weight loss and smoking cessation
> 4. Requiring attendance at weight loss and smoking cessation classes during work hours
> 5. Enforcing penalties for employees who fail to lose weight or stop smoking
> 6. **Reimbursing employees for expenses related to professional assistance for weight loss and smoking cessation** ✓
> 7. Creating and joining local community group meetings centered on weight loss and quitting smoking
> 8. Organizing a contest between branches to gradually reduce weight and smoking rates

### How to augment

Augmenting is done in `augment.py`. This reads in `train-orig.csv` and outputs `train.csv`, with augmented additions. To run the code, note that an OPENROUTER `API_KEY` must be supplied to enable model queries to be sent successfully. This can be set near the top of the file.